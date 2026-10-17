import glob
import os
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
    dataset = SentinelWaterDataset(root_dir="data/processed")
    print(f"dziala")