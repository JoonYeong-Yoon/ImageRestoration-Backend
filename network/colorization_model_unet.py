import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import logging
from skimage import color
from .models.util import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """모델 로드 중 발생하는 예외를 처리하기 위한 커스텀 예외 클래스"""
    pass

class ColorizationUNetModel:
    def __init__(self):
        """ResNet34 기반 UNet 컬러화 모델 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"UNet 컬러화 모델 초기화 중... 사용 중인 디바이스: {self.device}")

        try:
            # 모델 정의
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,  # 사전학습 안 함
                in_channels=1,
                classes=2
            ).to(self.device)

            # 가중치 로드
            weight_path = "network/weights/Colorization/best-smp-model-epoch=14-val_loss=0.01.ckpt"
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_path}")

            checkpoint = torch.load(weight_path, map_location=self.device)

            # PyTorch Lightning checkpoint 구조 대응
            if "state_dict" in checkpoint:
                state_dict = {
                    k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
                }
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

            logger.info("UNet 컬러화 모델 로드 완료 ✅")

        except Exception as e:
            error_msg = f"UNet 컬러화 모델 초기화 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        
    def colorize_with_unet(self, pil_data):
        """UNet을 사용하여 흑백 이미지를 컬러로 변환합니다."""
        if not self.model:
            raise ModelLoadError("UNet 모델이 로드되지 않았습니다")
        return self._process_image(pil_data)

    def _process_image(self, pil_data):
        """UNet 모델로 컬러화"""
        try:
            logger.info(f"UNet 이미지 처리 중: {pil_data}")
            
            # 원본 크기 저장
            original_size = pil_data.size  # (width, height)

            # 1️⃣ PIL → RGB → numpy → LAB
            img_rgb = pil_data.convert("RGB").resize((224, 224))
            img_np = np.array(img_rgb).astype(np.float32) / 255.0
            lab = color.rgb2lab(img_np)
            img_l = lab[:, :, 0:1] / 100.0  # [0,1] 범위

            # 2️⃣ L → tensor
            tens_l = torch.from_numpy(img_l).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # 3️⃣ 예측
            with torch.no_grad():
                out_ab = self.model(tens_l)

            # 4️⃣ 후처리: [0,1] → [-128,128]
            out_ab = out_ab.squeeze(0).cpu().permute(1, 2, 0).numpy()
            out_ab = (out_ab - 0.5) * 256.0 - 128.0  # 중심 맞추기
            lab_out = np.concatenate((lab[:, :, 0:1], out_ab), axis=2)
            lab_out[:, :, 0] = np.clip(lab_out[:, :, 0], 0, 100)
            lab_out[:, :, 1:] = np.clip(lab_out[:, :, 1:], -128, 127)

            # 5️⃣ LAB → RGB
            rgb_out = np.clip(color.lab2rgb(lab_out), 0, 1)
            out_img = Image.fromarray((rgb_out * 255).astype(np.uint8))
            
            out_img = out_img.resize(original_size, Image.BICUBIC)
            
            return out_img
        except Exception as e:
            error_msg = f"UNet 이미지 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise
