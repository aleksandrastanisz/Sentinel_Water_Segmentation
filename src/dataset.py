import glob
import os
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.indices import stack_multispectral_tensor


class SentinelWaterDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data/processed",
        transform: Optional[Callable] = None,
        scale_factor: float = 10000.0
    ):
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.transform = transform
        self.scale_factor = scale_factor
        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.tif")))

        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.npy")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_dir, filename)

        if img_path.endswith(".npy"): # tensor num
            raw_bands = np.load(img_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.uint8)
            b02, b03, b04, b08, b11 = [raw_bands[..., i] for i in range(5)]
        else: # geo tiff
            with rasterio.open(img_path) as src:
                bands = src.read().astype(np.float32) / self.scale_factor
                b02, b03, b04, b08, b11 = bands[0], bands[1], bands[2], bands[3], bands[4]

            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.uint8)

        features = stack_multispectral_tensor(b02, b03, b04, b08, b11)

        if self.transform is not None:
            augmented = self.transform(image=features, mask=mask)
            features = augmented["image"]
            mask = augmented["mask"]

        features_tensor = torch.from_numpy(features).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)

        return features_tensor, mask_tensor


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    images_dir = processed_dir / "images"
    masks_dir = processed_dir / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    h, w = 256, 256
    for i in range(2):

        dummy_bands = np.random.uniform(0.1, 0.4, (h, w, 5)).astype(np.float32)
        dummy_mask = np.zeros((h, w), dtype=np.uint8)

        dummy_bands[50:120, :, 3] = 0.02
        dummy_bands[50:120, :, 1] = 0.25  # Green
        dummy_mask[50:120, :] = 1

        np.save(images_dir / f"sample_{i}.npy", dummy_bands)
        np.save(masks_dir / f"sample_{i}.npy", dummy_mask)

    dataset = SentinelWaterDataset(root_dir=str(processed_dir))
    print(f"Liczba probek w processed: {len(dataset)}")

    features, mask = dataset[0]
    print(f"Tensor wejściowy: {features.shape}")
    print(f"Maska: {mask.shape}")