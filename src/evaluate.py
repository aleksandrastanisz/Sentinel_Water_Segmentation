from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader

from dataset import SentinelWaterDataset
from model import get_unet_model


def evaluate_and_plot(eps: float = 1e-7):
    project_root = Path(__file__).resolve().parent.parent
    val_dir = project_root / "data" / "processed" / "val"
    checkpoint_path = project_root / "outputs" / "checkpoints" / "unet_sentinel_water.pth"
    figures_dir = project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_unet_model(in_channels=7, num_classes=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    val_dataset = SentinelWaterDataset(root_dir=str(val_dir))
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    unet_inter, unet_union, unet_sum = 0, 0, 0
    otsu_inter, otsu_union, otsu_sum = 0, 0, 0

    vis_data = None

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(device)
            target_bool = masks.squeeze().numpy().astype(bool)

            outputs = model(images)
            preds_prob = torch.sigmoid(outputs).squeeze().cpu().numpy()
            unet_bool = preds_prob > 0.5

            ndwi_channel = images[0, 5].cpu().numpy()
            try:
                otsu_thresh = threshold_otsu(ndwi_channel)
                otsu_bool = ndwi_channel > otsu_thresh
            except ValueError:
                otsu_bool = np.zeros_like(ndwi_channel, dtype=bool)

            # Akumulacja globalna dla U-Net
            unet_inter += np.logical_and(unet_bool, target_bool).sum()
            unet_union += np.logical_or(unet_bool, target_bool).sum()
            unet_sum += unet_bool.sum() + target_bool.sum()

            # Akumulacja globalna dla Otsu
            otsu_inter += np.logical_and(otsu_bool, target_bool).sum()
            otsu_union += np.logical_or(otsu_bool, target_bool).sum()
            otsu_sum += otsu_bool.sum() + target_bool.sum()

            if i == 0:
                rgb = images[0, [2, 1, 0]].cpu().permute(1, 2, 0).numpy()
                p98 = np.percentile(rgb, 98)
                rgb = np.clip(rgb / (p98 if p98 > 0 else 1.0), 0, 1)
                vis_data = (
                    rgb,
                    otsu_bool.astype(np.uint8),
                    unet_bool.astype(np.uint8),
                    target_bool.astype(np.uint8)
                )

    unet_iou = (unet_inter + eps) / (unet_union + eps)
    unet_dice = (2.0 * unet_inter + eps) / (unet_sum + eps)

    otsu_iou = (otsu_inter + eps) / (otsu_union + eps)
    otsu_dice = (2.0 * otsu_inter + eps) / (otsu_sum + eps)

    print(f"Liczba próbek walidacyjnych: {len(loader)}")
    print(f"Baseline Otsu | Global IoU: {otsu_iou:.4f} | Global Dice: {otsu_dice:.4f}")
    print(f"Model U-Net   | Global IoU: {unet_iou:.4f} | Global Dice: {unet_dice:.4f}")

    if vis_data is not None:
        rgb, otsu_bin, unet_bin, gt = vis_data
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(rgb)
        axes[0].set_title("Kompozycja RGB")
        axes[0].axis("off")

        axes[1].imshow(otsu_bin, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Baseline: Otsu (NDWI)")
        axes[1].axis("off")

        axes[2].imshow(unet_bin, cmap="Blues", vmin=0, vmax=1)
        axes[2].set_title("Predykcja: U-Net (7-ch)")
        axes[2].axis("off")

        axes[3].imshow(gt, cmap="Blues", vmin=0, vmax=1)
        axes[3].set_title("Maska referencyjna (SCL)")
        axes[3].axis("off")

        plt.tight_layout()
        save_plot_path = figures_dir / "evaluation_results.png"
        plt.savefig(save_plot_path, dpi=200)
        plt.close()
        print(f"Zapisano wykres: {save_plot_path}")


if __name__ == "__main__":
    evaluate_and_plot()