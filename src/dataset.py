import glob
import os
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from indices import stack_multispectral_tensor


class SentinelWaterDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data/processed",
        transform: Optional[Callable] = None,
    ):
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.transform = transform
        # Kafelki zapisane przez download_data.py: pasma juz przeskalowane do 0-1
        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.npy")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_dir, filename)

        raw_bands = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)
        b02, b03, b04, b08, b11 = [raw_bands[..., i] for i in range(5)]

        features = stack_multispectral_tensor(b02, b03, b04, b08, b11)

        if self.transform is not None:
            augmented = self.transform(image=features, mask=mask)
            features = np.ascontiguousarray(augmented["image"])
            mask = np.ascontiguousarray(augmented["mask"])

        features_tensor = torch.from_numpy(features).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)

        return features_tensor, mask_tensor


if __name__ == "__main__":
    # Szybki podglad: wczytuje istniejace kafelki treningowe, niczego nie zapisuje
    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / "data" / "processed" / "train"

    dataset = SentinelWaterDataset(root_dir=str(train_dir))
    print(f"Liczba probek w {train_dir}: {len(dataset)}")

    if len(dataset) > 0:
        features, mask = dataset[0]
        print(f"Tensor wejsciowy: {tuple(features.shape)}")
        print(f"Maska: {tuple(mask.shape)}")
    else:
        print("Brak danych - uruchom najpierw src/download_data.py")
