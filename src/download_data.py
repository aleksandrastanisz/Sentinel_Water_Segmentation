from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from pystac_client import Client
import planetary_computer as pc


def fetch_and_tile_sentinel():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    images_dir = processed_dir / "images"
    masks_dir = processed_dir / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    bbox = [14.30, 53.60, 14.70, 53.90]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2023-06-01/2023-09-01",
        query={"eo:cloud_cover": {"lt": 5}},
        max_items=1
    )

    items = list(search.items())
    if not items:
        print("brak")
        return

    item = items[0]
    print(f"pobieranie dla: {item.id}")

    band_keys = ["B02", "B03", "B04", "B08", "B11"]
    bands_data = []

    win = Window(col_off=4000, row_off=4000, width=1024, height=1024)

    for key in band_keys:
        href = item.assets[key].href
        with rasterio.open(href) as src:
            data = src.read(1, window=win, out_shape=(1024, 1024)).astype(np.float32)
            data = np.clip(data / 10000.0, 0.0, 1.0)
            bands_data.append(data)

    stacked_bands = np.stack(bands_data, axis=-1)

    green = stacked_bands[:, :, 1]
    swir = stacked_bands[:, :, 4]
    mndwi = (green - swir) / (green + swir + 1e-7)
    water_mask = (mndwi > 0.0).astype(np.uint8)

    tile_size = 256
    saved_count = 0

    for r in range(0, 1024, tile_size):
        for c in range(0, 1024, tile_size):
            img_tile = stacked_bands[r:r+tile_size, c:c+tile_size, :]
            mask_tile = water_mask[r:r+tile_size, c:c+tile_size]

            water_ratio = mask_tile.mean()

            if 0.10 <= water_ratio <= 0.90:
                out_img = images_dir / f"water_tile_{saved_count}.npy"
                out_mask = masks_dir / f"water_tile_{saved_count}.npy"

                np.save(out_img, img_tile)
                np.save(out_mask, mask_tile)
                saved_count += 1

    print(f"wygenerowano {saved_count} zdjec w {processed_dir}")


if __name__ == "__main__":
    fetch_and_tile_sentinel()