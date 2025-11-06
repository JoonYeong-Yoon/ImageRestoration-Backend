import os
import torch
import logging
from PIL import Image
from torchvision import transforms
from network.models.unet_restore_model import UNetRestoreModel
import segmentation_models_pytorch as smp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """모델 로드 중 발생하는 예외"""
    pass


class RestoreUNetModel:
    """
    손상 이미지 복원을 위한 U-Net 기반 모델 핸들러
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🧩 UNet 복원 모델 초기화 중... (device={self.device})")

        try:
            # 모델 정의
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=3,
                activation='tanh'
            ).to("cpu")
            
            # 가중치 파일 경로
            weight_path = os.path.join("network", "weights", "damageRestoration", "last_epoch_model_epoch_3.pth")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"가중치 파일이 존재하지 않습니다: {weight_path}")

            # 체크포인트 로드
            checkpoint = torch.load(weight_path, map_location=self.device)
            if "generator_state_dict" in checkpoint:
                state_dict = checkpoint["generator_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            logger.info("✅ UNet 복원 모델 로드 완료")

        except Exception as e:
            msg = f"UNet 복원 모델 초기화 실패: {e}"
            logger.error(msg)
            raise ModelLoadError(msg)


    def restore_with_unet(self, pil_data: Image.Image) -> Image.Image:
        """U-Net을 이용한 이미지 복원"""
        if not self.model:
            raise ModelLoadError("UNet 복원 모델이 로드되지 않았습니다.")
        return self._process_image(pil_data)


    def _process_image(self, pil_data: Image.Image) -> Image.Image:
        """이미지 복원 파이프라인"""
        try:
            logger.info(f"복원 시작 - 이미지 크기: {pil_data.size}, 모드: {pil_data.mode}")
            original_size = pil_data.size

            # --- 전처리 ---
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )

            img_tensor = transform(pil_data).unsqueeze(0).to(self.device)

            # --- 추론 ---
            with torch.no_grad():
                restored_tensor = self.model(img_tensor)

            # --- 후처리 ---
            restored_tensor = inv_normalize(restored_tensor.squeeze(0).cpu())
            restored_tensor = torch.clamp(restored_tensor, 0, 1)
            restored_img = transforms.ToPILImage()(restored_tensor)
            restored_img = restored_img.resize(original_size, Image.BICUBIC)

            logger.info("✅ 복원 완료")
            return restored_img

        except Exception as e:
            msg = f"UNet 복원 중 오류 발생: {e}"
            logger.error(msg)
            raise
