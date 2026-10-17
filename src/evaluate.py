from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader

from dataset import SentinelWaterDataset
from model import get_unet_model

def compute_iou_and_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    total = np.logical_or(pred, target).sum()
    union_sum = pred.sum() + target.sum()

    iou = (intersection + eps) / (total + eps)
    dice = (2.0 * intersection + eps) / (union_sum + eps)
    return float(iou), float(dice)

def evaluate_and_plot():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    checkpoint_path = project_root / "outputs" / "checkpoints" / "unet_sentinel_water.pth"
    figures_dir = project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_unet_model(in_channels=7, num_classes=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = SentinelWaterDataset(root_dir=str(processed_dir))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_unet_iou, total_unet_dice = 0.0, 0.0
    total_otsu_iou, total_otsu_dice = 0.0, 0.0

    vis_data = None

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks_np = masks.squeeze().numpy().astype(np.uint8)

            outputs = model(images)
            preds_prob = torch.sigmoid(outputs).squeeze().cpu().numpy()
            preds_binary = (preds_prob > 0.5).astype(np.uint8)

            ndwi_channel = images[0, 5].cpu().numpy()
            try:
                otsu_thresh = threshold_otsu(ndwi_channel)
                otsu_binary = (ndwi_channel > otsu_thresh).astype(np.uint8)
            except ValueError:
                otsu_binary = np.zeros_like(ndwi_channel, dtype=np.uint8)

            # metryki
            u_iou, u_dice = compute_iou_and_dice(preds_binary, masks_np)
            o_iou, o_dice = compute_iou_and_dice(otsu_binary, masks_np)

            total_unet_iou += u_iou
            total_unet_dice += u_dice
            total_otsu_iou += o_iou
            total_otsu_dice += o_dice

            if i == 0:
                rgb = images[0, [2, 1, 0]].cpu().permute(1, 2, 0).numpy()
                rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)  # Normalizacja podglądu
                vis_data = (rgb, otsu_binary, preds_binary, masks_np)

    num_samples = len(loader)
    print(f"Liczba próbek: {num_samples}")
    print(f"Baseline Otsu | Średnie IoU: {total_otsu_iou / num_samples:.4f} | Średni Dice: {total_otsu_dice / num_samples:.4f}")
    print(f"Model U-Net   | Średnie IoU: {total_unet_iou / num_samples:.4f} | Średni Dice: {total_unet_dice / num_samples:.4f}")

    if vis_data is not None:
        rgb, otsu_bin, unet_bin, gt = vis_data
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(rgb)
        axes[0].set_title("Kompozycja RGB")
        axes[0].axis("off")

        axes[1].imshow(otsu_bin, cmap="Blues")
        axes[1].set_title("Baseline: Otsu (NDWI)")
        axes[1].axis("off")

        axes[2].imshow(unet_bin, cmap="Blues")
        axes[2].set_title("Predykcja: U-Net (7-ch)")
        axes[2].axis("off")

        axes[3].imshow(gt, cmap="Blues")
        axes[3].set_title("Maska referencyjna (GT)")
        axes[3].axis("off")

        plt.tight_layout()
        save_plot_path = figures_dir / "evaluation_results.png"
        plt.savefig(save_plot_path, dpi=200)
        plt.close()
        print(f"zapisano wykres: {save_plot_path}")


if __name__ == "__main__":
    evaluate_and_plot()