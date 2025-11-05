import os
import torch
import numpy as np
from PIL import Image
import logging
from torchvision import transforms
from network.models.unet_restore_model import UNetRestoreModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """모델 로드 중 발생하는 예외 처리용"""
    pass


class RestoreUNetModel:
    def __init__(self):
        """U-Net 기반 복원 모델 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🧩 UNet 복원 모델 초기화 중... 사용 디바이스: {self.device}")

        try:
            # 모델 정의
            self.model = UNetRestoreModel(
                encoder_name="resnet34",
                encoder_weights="imagenet",  # 필요 시 None으로 변경
                in_channels=3,
                classes=3
            ).to(self.device)

            # 가중치 로드
            weight_path = "network\weights\damageRestoration\generator_epoch_11_loss_9.0019.pth"
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_path}")

            checkpoint = torch.load(weight_path, map_location=self.device)
            if "state_dict" in checkpoint:
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

            logger.info("✅ UNet 복원 모델 로드 완료")

        except Exception as e:
            msg = f"UNet 복원 모델 초기화 중 오류 발생: {e}"
            logger.error(msg)
            raise ModelLoadError(msg)

    def restore_with_unet(self, pil_data):
        """손상 이미지를 복원"""
        if not self.model:
            raise ModelLoadError("UNet 복원 모델이 로드되지 않았습니다.")
        return self._process_image(pil_data)

    def _process_image(self, pil_data):
        try:
            logger.info(f"복원 중: {pil_data}")

            # 원본 크기 저장
            original_size = pil_data.size  # (W,H)
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            img_tensor = transform(pil_data).unsqueeze(0).to(self.device)

            with torch.no_grad():
                restored_tensor = self.model(img_tensor)

            restored_img = transforms.ToPILImage()(restored_tensor.squeeze(0).cpu())
            restored_img = restored_img.resize(original_size, Image.BICUBIC)

            return restored_img

        except Exception as e:
            msg = f"UNet 복원 중 오류 발생: {e}"
            logger.error(msg)
            raise
