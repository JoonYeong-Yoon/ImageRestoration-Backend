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

# ✅ 모델 클래스 정의
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


# ------------------------------------------------------------
# 환경 설정
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 경로 수정 (backend/checkpoints_unet 바로 아래에 파일이 있다고 가정)
MODEL_PATH = os.path.join("checkpoints_unet", "pj2-color.ckpt")

# ✅ 모델 로드
if os.path.exists(MODEL_PATH):
    print("🧠 AI 모델 로드 중...")
    lit_model = LitColorization.load_from_checkpoint(MODEL_PATH)
    lit_model.to(device)
    lit_model.eval()
    print("✅ AI 모델 준비 완료!")
else:
    print("⚠️ 모델 파일을 찾을 수 없습니다:", MODEL_PATH)
    lit_model = None


# ------------------------------------------------------------
# Transform 정의
# ------------------------------------------------------------
transform_gray = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])


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
