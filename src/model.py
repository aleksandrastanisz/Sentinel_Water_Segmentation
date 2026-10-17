import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_unet_model(
    in_channels: int = 7,
    num_classes: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet"
) -> nn.Module:
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None
    )
    return model

if __name__ == "__main__":
    # test
    model = get_unet_model(in_channels=7, num_classes=1)

    dummy_input = torch.randn(2, 7, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"in: {dummy_input.shape}")
    print(f"out: {output.shape}")