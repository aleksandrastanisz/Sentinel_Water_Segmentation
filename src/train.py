import os
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

def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    checkpoints_dir = project_root / "outputs" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trening na: {device}")

    dataset = SentinelWaterDataset(root_dir=str(processed_dir))
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = get_unet_model(in_channels=7, num_classes=1).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()

    def compound_loss(logits, targets):
        return 0.5 * bce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 3
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, compound_loss, optimizer, device)
        print(f"Epoka {epoch}/{epochs} | Loss: {loss:.4f}")

    weights_path = checkpoints_dir / "unet_sentinel_water.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"zapisane wagi: {weights_path}")

if __name__ == "__main__":
    main()