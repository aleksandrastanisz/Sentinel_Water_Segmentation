import shutil
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client
import planetary_computer as pc


def fetch_and_tile_sentinel():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    for split in ["train", "val"]:
        (processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (processed_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    bbox = [14.35, 53.65, 14.65, 53.85]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2023-06-01/2023-09-01",
        query={"eo:cloud_cover": {"lt": 5}},
        max_items=1
    )

    items = list(search.items())
    if not items:
        print("Brak scen spełniających kryteria.")
        return

    item = items[0]
    print(f"Pobieranie dla sceny: {item.id}")

    ref_href = item.assets["B02"].href
    with rasterio.open(ref_href) as ref_src:
        proj_bounds = transform_bounds("EPSG:4326", ref_src.crs, *bbox)
        ref_win = from_bounds(*proj_bounds, transform=ref_src.transform)
        ref_win = ref_win.round_lengths().round_offsets()
        target_h = int(ref_win.height)
        target_w = int(ref_win.width)

    print(f"Rozmiar wycinka w rozdzielczości 10m: {target_h}x{target_w} px")

    band_keys = ["B02", "B03", "B04", "B08", "B11"]
    bands_data = []

    for key in band_keys:
        href = item.assets[key].href
        with rasterio.open(href) as src:
            band_win = from_bounds(*proj_bounds, transform=src.transform)
            band_win = band_win.round_lengths().round_offsets()

            resamp = Resampling.bilinear if key == "B11" else Resampling.nearest

            raw = src.read(
                1,
                window=band_win,
                out_shape=(target_h, target_w),
                resampling=resamp
            ).astype(np.float32)

            if np.percentile(raw[raw > 0], 1) > 800:
                raw = raw - 1000.0

            data = np.clip(raw / 10000.0, 0.0, 1.0)
            bands_data.append(data)

    stacked_bands = np.stack(bands_data, axis=-1)

    scl_href = item.assets["SCL"].href
    with rasterio.open(scl_href) as scl_src:
        scl_win = from_bounds(*proj_bounds, transform=scl_src.transform)
        scl_win = scl_win.round_lengths().round_offsets()

        scl_data = scl_src.read(
            1,
            window=scl_win,
            out_shape=(target_h, target_w),
            resampling=Resampling.nearest
        )

    water_mask = (scl_data == 6).astype(np.uint8)
    invalid_mask = np.isin(scl_data, [0, 3, 8, 9, 10])

    tile_size = 256
    valid_tiles = []

    for r in range(0, target_h - tile_size + 1, tile_size):
        for c in range(0, target_w - tile_size + 1, tile_size):
            img_tile = stacked_bands[r:r + tile_size, c:c + tile_size, :]
            mask_tile = water_mask[r:r + tile_size, c:c + tile_size]
            inv_tile = invalid_mask[r:r + tile_size, c:c + tile_size]

            if inv_tile.mean() > 0.05:
                continue

            water_ratio = float(mask_tile.mean())
            if 0.02 <= water_ratio <= 0.98:
                valid_tiles.append((img_tile, mask_tile))

    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(valid_tiles))
    split_idx = int(0.8 * len(valid_tiles))

    train_count, val_count = 0, 0
    for idx_pos, tile_idx in enumerate(indices):
        img_tile, mask_tile = valid_tiles[tile_idx]
        split = "train" if idx_pos < split_idx else "val"

        out_img = processed_dir / split / "images" / f"water_tile_{tile_idx:04d}.npy"
        out_mask = processed_dir / split / "masks" / f"water_tile_{tile_idx:04d}.npy"

        np.save(out_img, img_tile)
        np.save(out_mask, mask_tile)

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    print(f"Zapisano {train_count} kafelków w 'train' oraz {val_count} w 'val' (katalog: {processed_dir})")


if __name__ == "__main__":
    fetch_and_tile_sentinel()