import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from skimage import color

# ===============================
# 1️⃣ AI 담당자 UNet 모델 정의
# ===============================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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

        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()  # 출력 범위 [-1, 1]

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
        out = self.activation(out)
        return out

# ===============================
# 2️⃣ 모델 로드 함수
# ===============================
def load_colorizer(model_path):
    model = UNet(in_channels=1, out_channels=2)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✅ UNet 모델 로드 완료: {model_path}")
    else:
        print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
    model.eval()
    return model

# ===============================
# 3️⃣ 이미지 컬러화 함수 (학습 정규화 반영)
# ===============================
def colorize_image(model, input_path, output_dir):
    # L 채널 흑백 이미지
    image = Image.open(input_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((320, 320)),           # 학습 시 target_size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[1.0])  # 이미 L은 [-1,1]로 후처리 예정
    ])
    l_tensor = transform(image).unsqueeze(0)
    
    # 학습 시 정규화: L [-1,1] 범위, 모델 입력 그대로
    l_tensor = (np.array(image.resize((320,320)), dtype=np.float32) / 50.0) - 1.0
    l_tensor = torch.from_numpy(l_tensor).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    with torch.no_grad():
        ab_output = model(l_tensor)  # UNet 출력 ab 채널 [-1,1]
        ab_output = ab_output[0].permute(1, 2, 0).cpu().numpy()
        ab_output = np.clip(ab_output, -1, 1)  # 안정화
        ab_output = ab_output * 128            # [-128,128] 범위

    # L 채널 복원: 학습 시 정규화와 반대로 처리
    l_channel = ((l_tensor[0,0].cpu().numpy() + 1.0) * 50.0)  # 0~100

    # Lab 이미지 합치기
    lab_image = np.zeros((320, 320, 3), dtype=np.float32)
    lab_image[..., 0] = l_channel
    lab_image[..., 1:] = ab_output

    # Lab -> RGB
    rgb_image = color.lab2rgb(lab_image)
    rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
    output_image = Image.fromarray(rgb_image)

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"colorized_{os.path.basename(input_path)}")
    output_image.save(output_path)
    print(f"🎨 컬러화 완료: {output_path}")
    return output_path
