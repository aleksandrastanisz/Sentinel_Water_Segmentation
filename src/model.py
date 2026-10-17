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