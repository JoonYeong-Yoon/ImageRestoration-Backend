import os
import torch
import torch.nn as nn
from PIL import Image
import logging
import torchvision.transforms as transforms
from .models.uformer import Uformer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    pass

class RestorationModel:
    def __init__(self, checkpoint_path):
        logger.info("복원 모델 초기화 중...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 중인 디바이스: {self.device}")
        
        try:
            # 모델 초기화
            logger.info("Uformer 모델 인스턴스 생성 중...")
            self.model = Uformer().to(self.device)
            
            # 체크포인트 파일 존재 확인
            if not os.path.exists(checkpoint_path):
                error_msg = f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg)
            
            # 체크포인트 로드
            logger.info(f"체크포인트 로드 중: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("체크포인트에서 model_state_dict를 성공적으로 로드했습니다.")
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info("체크포인트에서 모델 가중치를 성공적으로 로드했습니다.")
            except Exception as e:
                error_msg = f"체크포인트 로드 실패: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg)
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            logger.info("모델이 성공적으로 초기화되었고 평가 모드로 설정되었습니다.")
        except Exception as e:
            error_msg = f"모델 초기화 중 예상치 못한 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        
        # 이미지 변환 설정
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.ToPILImage()
        ])
    
    def process_image(self, image_path):
        try:
            # 이미지 파일 존재 확인
            if not os.path.exists(image_path):
                error_msg = f"이미지 파일을 찾을 수 없습니다: {image_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 이미지 로드 및 전처리
            logger.info(f"이미지 처리 중: {image_path}")
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 이미지 처리
            logger.info("이미지 복원 처리 중...")
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # 후처리 및 반환
            restored_image = self.inverse_transform(output_tensor.squeeze().cpu())
            logger.info("이미지 처리가 성공적으로 완료되었습니다.")
            return restored_image
            
        except Exception as e:
            error_msg = f"이미지 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise