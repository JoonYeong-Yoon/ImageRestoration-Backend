import segmentation_models_pytorch as smp
import torch.nn as nn

class UNetRestoreModel(smp.Unet):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=3):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
