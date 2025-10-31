from PIL import Image
import os

# ------------------------------------------------------------
# 기본 화질 개선 함수 (단순 업스케일)
# ------------------------------------------------------------
PROCESSED_FOLDER = os.path.join(os.getcwd(), "backend", "processed")
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def enhance_image(input_path):
    """화질 개선 (임시: 단순 업스케일)"""
    img = Image.open(input_path)
    enhanced = img.resize((img.width * 2, img.height * 2))
    
    output_path = os.path.join(PROCESSED_FOLDER, "enhanced_" + os.path.basename(input_path))
    enhanced.save(output_path, quality=95)
    return output_path


# ------------------------------------------------------------
# AI 컬러화 기능 (UNet 기반)
# ------------------------------------------------------------
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance
from torch import nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from backend.backend.ImageRestoration_Backend.models.colorization.eccv16 import eccv16

# ------------------------------------------------------------
# 환경 설정
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Transform 정의
# ------------------------------------------------------------
transform_gray = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

from backend.backend.ImageRestoration_Backend.models.colorizer import colorize_image

# ------------------------------------------------------------
# AI 컬러화 함수
# ------------------------------------------------------------
def colorize_image(file):
    """흑백 이미지를 AI로 컬러 복원"""
    if lit_model is None:
        raise FileNotFoundError("모델이 로드되지 않았습니다.")

    img = Image.open(file.stream).convert("L")
    orig_size = img.size
    x = transform_gray(img).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = lit_model(x)
        y_pred = (y_pred + 1) / 2
        y_pred = torch.clamp(y_pred, 0, 1)

    # 후처리 및 저장
    y_pred_pil = T.ToPILImage()(y_pred.squeeze(0).cpu())
    y_pred_pil = y_pred_pil.resize(orig_size, Image.BICUBIC)
    y_pred_pil = ImageEnhance.Sharpness(y_pred_pil).enhance(1.5)

    output_path = os.path.join(PROCESSED_FOLDER, "colorized_result.png")
    y_pred_pil.save(output_path)
    return output_path
