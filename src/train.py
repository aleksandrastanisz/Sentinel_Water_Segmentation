from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import SentinelWaterDataset
from model import get_unet_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, eps=1e-7):
    model.eval()
    total_loss = 0.0
    total_inter = 0.0
    total_union = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks_float = masks.to(device).float()

        outputs = model(images)
        loss = criterion(outputs, masks_float)
        total_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).long()
        targets = masks.to(device).long()

        inter = (preds & targets).sum().item()
        union = (preds | targets).sum().item()

        total_inter += inter
        total_union += union

    avg_loss = total_loss / len(loader)
    global_iou = (total_inter + eps) / (total_union + eps)
    return avg_loss, global_iou


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    checkpoints_dir = project_root / "outputs" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trening na: {device}")

    train_dataset = SentinelWaterDataset(root_dir=str(processed_dir / "train"))
    val_dataset = SentinelWaterDataset(root_dir=str(processed_dir / "val"))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = get_unet_model(in_channels=7, num_classes=1).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()

    def compound_loss(logits, targets):
        return 0.5 * bce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 25
    best_val_iou = 0.0
    weights_path = checkpoints_dir / "unet_sentinel_water.pth"

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, compound_loss, optimizer, device)
        val_loss, val_iou = validate_epoch(model, val_loader, compound_loss, device)

        print(
            f"Epoka {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), weights_path)

    print(f"Zakończono. Najlepsze Val IoU: {best_val_iou:.4f} (zapisano w: {weights_path})")


if __name__ == "__main__":
    main()