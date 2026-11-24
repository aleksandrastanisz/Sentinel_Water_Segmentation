import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


@dataclass
class Region:
    name: str
    bbox: list  # [min_lon, min_lat, max_lon, max_lat] w EPSG:4326
    split: str  # "train", "val" albo "test"


# Kazdy obszar trafia w calosci do jednego zbioru, wiec ten sam zbiornik
# nigdy nie pojawia sie jednoczesnie w treningu i w ewaluacji.
REGIONS = [
    Region("szczecin_lagoon", [14.35, 53.65, 14.65, 53.85], "train"),
    Region("sniardwy_lake", [21.55, 53.70, 21.85, 53.82], "train"),
    Region("vistula_lagoon", [19.40, 54.30, 19.70, 54.42], "train"),
    Region("wloclawek_reservoir", [18.90, 52.58, 19.20, 52.72], "train"),
    Region("goplo_lake", [18.30, 52.50, 18.45, 52.70], "val"),
    Region("jeziorsko_reservoir", [18.60, 51.70, 18.78, 51.85], "test"),
    Region("solina_reservoir", [22.38, 49.33, 22.55, 49.45], "test"),
]

DATE_RANGE = "2023-06-01/2023-09-01"
MAX_CLOUD_COVER = 10
BAND_KEYS = ["B02", "B03", "B04", "B08", "B11"]
TILE_SIZE = 256
SPLITS = ["train", "val", "test"]


def find_least_cloudy_scene(catalog, bbox):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        max_items=20,
    )
    items = list(search.items())
    if not items:
        return None
    return min(items, key=lambda it: it.properties["eo:cloud_cover"])


def get_offset(item):
    # Od Processing Baseline 04.00 (styczen 2022) wartosci L2A maja przesuniecie +1000
    baseline = item.properties.get("s2:processing_baseline")
    if baseline is None:
        print("  Brak informacji o processing baseline - offset wykrywany heurystycznie")
        return None
    offset = -1000.0 if float(baseline) >= 4.0 else 0.0
    print(f"  Processing baseline: {baseline}, offset: {offset:.0f}")
    return offset


def read_bands(item, bbox):
    with rasterio.open(item.assets["B02"].href) as ref_src:
        proj_bounds = transform_bounds("EPSG:4326", ref_src.crs, *bbox)
        ref_win = from_bounds(*proj_bounds, transform=ref_src.transform)
        ref_win = ref_win.round_lengths().round_offsets()
        target_h, target_w = int(ref_win.height), int(ref_win.width)

    offset = get_offset(item)
    bands_data = []
    for key in BAND_KEYS:
        with rasterio.open(item.assets[key].href) as src:
            band_win = from_bounds(*proj_bounds, transform=src.transform)
            band_win = band_win.round_lengths().round_offsets()
            resamp = Resampling.bilinear if key == "B11" else Resampling.nearest

            raw = src.read(
                1, window=band_win, out_shape=(target_h, target_w), resampling=resamp
            ).astype(np.float32)

            if offset is not None:
                raw = raw + offset
            elif np.percentile(raw[raw > 0], 1) > 800:
                raw = raw - 1000.0

            bands_data.append(np.clip(raw / 10000.0, 0.0, 1.0))

    with rasterio.open(item.assets["SCL"].href) as scl_src:
        scl_win = from_bounds(*proj_bounds, transform=scl_src.transform)
        scl_win = scl_win.round_lengths().round_offsets()
        scl = scl_src.read(
            1, window=scl_win, out_shape=(target_h, target_w), resampling=Resampling.nearest
        )

    return np.stack(bands_data, axis=-1), scl


def save_tiles(region, bands, scl, processed_dir):
    water_mask = (scl == 6).astype(np.uint8)
    invalid_mask = np.isin(scl, [0, 3, 8, 9, 10])  # brak danych, cien chmur, chmury, cirrus

    h, w = water_mask.shape
    saved = 0
    for r in range(0, h - TILE_SIZE + 1, TILE_SIZE):
        for c in range(0, w - TILE_SIZE + 1, TILE_SIZE):
            if invalid_mask[r:r + TILE_SIZE, c:c + TILE_SIZE].mean() > 0.05:
                continue
            mask_tile = water_mask[r:r + TILE_SIZE, c:c + TILE_SIZE]
            if not 0.02 <= float(mask_tile.mean()) <= 0.98:
                continue

            name = f"{region.name}_{r:05d}_{c:05d}.npy"
            out_dir = processed_dir / region.split
            np.save(out_dir / "images" / name, bands[r:r + TILE_SIZE, c:c + TILE_SIZE, :])
            np.save(out_dir / "masks" / name, mask_tile)
            saved += 1
    return saved


def fetch_and_tile_sentinel():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    for split in SPLITS:
        (processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (processed_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    counts = {split: 0 for split in SPLITS}
    for region in REGIONS:
        print(f"[{region.split}] {region.name}")
        item = find_least_cloudy_scene(catalog, region.bbox)
        if item is None:
            print("  Brak scen spelniajacych kryteria - pomijam")
            continue
        print(f"  Scena: {item.id} (zachmurzenie {item.properties['eo:cloud_cover']:.1f}%)")

        bands, scl = read_bands(item, region.bbox)
        saved = save_tiles(region, bands, scl, processed_dir)
        counts[region.split] += saved
        print(f"  Zapisano kafelkow: {saved}")

    print("Podsumowanie: " + ", ".join(f"{s}: {n}" for s, n in counts.items()))


if __name__ == "__main__":
    fetch_and_tile_sentinel()
