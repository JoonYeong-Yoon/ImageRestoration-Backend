"""
train_colorization_unet.py
- 흑백 → 컬러화 UNet 학습 스크립트
- 학습 완료 시 checkpoints_unet/ 폴더에 .ckpt 파일 자동 저장
"""

import os
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import segmentation_models_pytorch as smp
from torchvision.utils import save_image


# ------------------------------------------------------------
# ✅ 데이터셋 클래스 정의
# ------------------------------------------------------------
class ColorizationDataset(Dataset):
    def __init__(self, image_dir, size=128):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.transform_rgb = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()
        ])
        self.transform_gray = T.Compose([
            T.Resize((size, size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        y = self.transform_rgb(img)
        x = self.transform_gray(img)
        return x, y


# ------------------------------------------------------------
# ✅ Lightning 모델 정의
# ------------------------------------------------------------
class LitColorization(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=3
        )
        self.loss_fn = nn.L1Loss()
        self.lr = lr
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


# ------------------------------------------------------------
# ✅ 메인 실행 부분
# ------------------------------------------------------------
if __name__ == "__main__":
    # 학습용 이미지 폴더 지정
    train_dir = r"E:\image_train"      # ⚠️ 너의 이미지 데이터 경로로 변경
    val_dir = r"E:\image_val"          # ⚠️ 검증용 데이터 경로로 변경

    # 폴더 존재 확인
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"학습용 폴더가 없습니다: {train_dir}")

    os.makedirs("checkpoints_unet", exist_ok=True)

    # 데이터셋 생성
    train_dataset = ColorizationDataset(train_dir)
    val_dataset = ColorizationDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # 모델 초기화
    model = LitColorization()

    # 콜백 설정
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_unet",
        filename="unet-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    # 학습 실행
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        precision=32,
        callbacks=[checkpoint_cb, early_stop],
        log_every_n_steps=10
    )

    print("🧠 모델 학습 시작...")
    trainer.fit(model, train_loader, val_loader)
    print("✅ 학습 완료! 체크포인트 파일이 checkpoints_unet 폴더에 생성되었습니다.")
