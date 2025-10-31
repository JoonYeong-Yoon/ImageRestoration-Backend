import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import os

# 간단한 컬러화 모델 예시
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ 모델 로드 함수
def load_colorizer(model_path):
    model = ColorizationNet()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        print(f"✅ 팀장님 모델 로드 완료: {model_path}")
    else:
        print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")

    model.eval()
    return model

# ✅ 실제 이미지 컬러화 함수
def colorize_image(model, input_path, output_dir):
    image = Image.open(input_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        output = output[0].permute(1, 2, 0).numpy()
    print(output.max())
    output = (output * 255).clip(0, 255).astype(np.uint8)
    print(output.shape, output.max())
    output_image = Image.fromarray(output)

    output_path = os.path.join(output_dir, f"colorized_{os.path.basename(input_path)}")
    output_image.save(output_path)

    print(f"🎨 컬러화 완료: {output_path}")
    return output_path
