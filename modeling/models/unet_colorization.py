
import torch
import torch.nn as nn
import pytorch_lightning as pl

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_c=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_c)
        self.enc2 = DoubleConv(base_c, base_c*2)
        self.enc3 = DoubleConv(base_c*2, base_c*4)
        self.enc4 = DoubleConv(base_c*4, base_c*8)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = DoubleConv(base_c*8 + base_c*4, base_c*4)
        self.dec2 = DoubleConv(base_c*4 + base_c*2, base_c*2)
        self.dec1 = DoubleConv(base_c*2 + base_c, base_c)
        self.final_conv = nn.Conv2d(base_c, out_channels, 3, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final_conv(d1)
        return self.activation(out)

class LitColorization(pl.LightningModule):
    def __init__(self, lr=1e-4, base_c=32):
        super().__init__()
        self.model = UNet(in_channels=1, out_channels=2, base_c=base_c)
        self.loss_fn = nn.L1Loss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        L, ab = batch
        pred_ab = self(L)
        loss = self.loss_fn(pred_ab, ab)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        L, ab = batch
        pred_ab = self(L)
        loss = self.loss_fn(pred_ab, ab)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
