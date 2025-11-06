import segmentation_models_pytorch as smp
import torch.nn as nn

class UNetRestoreModel(nn.Module):
    """
    손상 이미지 복원을 위한 U-Net 기반 모델
    """
    def __init__(self, encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3, activation='tanh'):
        super(UNetRestoreModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )

    def forward(self, x):
        return self.model(x)
