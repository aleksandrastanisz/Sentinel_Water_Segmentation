import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader

from dataset import SentinelWaterDataset
from model import get_unet_model

METHODS = ["Otsu (NDWI)", "Otsu (MNDWI)", "U-Net"]


def region_from_path(path: str) -> str:
    # "solina_reservoir_00256_00512.npy" -> "solina_reservoir"
    return Path(path).stem.rsplit("_", 2)[0]


def otsu_mask(index: np.ndarray) -> np.ndarray:
    try:
        return index > threshold_otsu(index)
    except ValueError:  # staly obraz, brak progu
        return np.zeros_like(index, dtype=bool)


def compute_metrics(c: dict, eps: float = 1e-7) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "IoU": (tp + eps) / (tp + fp + fn + eps),
        "Dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "Precision": (tp + eps) / (tp + fp + eps),
        "Recall": (tp + eps) / (tp + fn + eps),
    }


def evaluate_and_plot(split: str = "test"):
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed" / split
    checkpoint_path = project_root / "outputs" / "checkpoints" / "unet_sentinel_water.pth"
    figures_dir = project_root / "outputs" / "figures"
    tables_dir = project_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_unet_model(in_channels=7, num_classes=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    dataset = SentinelWaterDataset(root_dir=str(data_dir))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # counts[region][method] = {"tp", "fp", "fn"}; region "ALL" = wszystkie kafelki
    counts = defaultdict(lambda: {m: {"tp": 0, "fp": 0, "fn": 0} for m in METHODS})
    examples = {}  # pierwszy kafelek z kazdego obszaru do wykresu

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            region = region_from_path(dataset.image_paths[i])
            target = masks.squeeze().numpy().astype(bool)

            logits = model(images.to(device))
            preds = {
                "U-Net": torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5,
                "Otsu (NDWI)": otsu_mask(images[0, 5].numpy()),  # kanal 5 = NDWI
                "Otsu (MNDWI)": otsu_mask(images[0, 6].numpy()),  # kanal 6 = MNDWI
            }

            for method, pred in preds.items():
                for key in (region, "ALL"):
                    c = counts[key][method]
                    c["tp"] += int(np.logical_and(pred, target).sum())
                    c["fp"] += int(np.logical_and(pred, ~target).sum())
                    c["fn"] += int(np.logical_and(~pred, target).sum())

            if region not in examples:
                rgb = images[0, [2, 1, 0]].permute(1, 2, 0).numpy()
                p98 = np.percentile(rgb, 98)
                rgb = np.clip(rgb / (p98 if p98 > 0 else 1.0), 0, 1)
                examples[region] = (rgb, preds["Otsu (NDWI)"], preds["Otsu (MNDWI)"], preds["U-Net"], target)

    # Tabela wynikow: osobno kazdy obszar + wszystkie razem
    regions = sorted(r for r in counts if r != "ALL") + ["ALL"]
    rows = []
    print(f"Zbior: {split}, liczba kafelkow: {len(dataset)}")
    print(f"{'Obszar':<22}{'Metoda':<15}{'IoU':>8}{'Dice':>8}{'Prec':>8}{'Recall':>8}")
    for region in regions:
        for method in METHODS:
            m = compute_metrics(counts[region][method])
            rows.append({"Region": region, "Method": method, **{k: f"{v:.4f}" for k, v in m.items()}})
            print(f"{region:<22}{method:<15}{m['IoU']:>8.4f}{m['Dice']:>8.4f}"
                  f"{m['Precision']:>8.4f}{m['Recall']:>8.4f}")

    csv_path = tables_dir / f"metrics_{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Zapisano tabele: {csv_path}")

    # Wykres: jeden wiersz na obszar
    titles = ["RGB", "Otsu (NDWI)", "Otsu (MNDWI)", "U-Net (7 channels)", "Reference (SCL)"]
    fig, axes = plt.subplots(len(examples), 5, figsize=(20, 4 * len(examples)), squeeze=False)
    for row, (region, images_row) in enumerate(sorted(examples.items())):
        for col, img in enumerate(images_row):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(img)
                ax.set_ylabel(region)
            else:
                ax.imshow(img.astype(np.uint8), cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col])

    plt.tight_layout()
    fig_path = figures_dir / f"evaluation_{split}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres: {fig_path}")


if __name__ == "__main__":
    evaluate_and_plot(split="test")
