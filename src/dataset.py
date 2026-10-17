import glob
import os
from typing import Optional, Callable

from torch.utils.data import Dataset


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

if __name__ == "__main__":
    dataset = SentinelWaterDataset(root_dir="data/processed")
    print(f"dziala")