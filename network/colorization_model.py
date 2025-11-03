import os
import logging
from PIL import Image
import torch
import numpy as np
from .models.eccv16 import eccv16
from .models.util import *

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """모델 로드 중 발생하는 예외를 처리하기 위한 커스텀 예외 클래스"""
    pass

class ColorizationModel:
    def __init__(self):
        """컬러화 모델을 초기화합니다."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"컬러화 모델 초기화 중... 사용 중인 디바이스: {self.device}")
        
        try:
            # ECCV16 모델 로드
            logger.info("ECCV16 모델 로드 중...")
            self.eccv16_model = eccv16(pretrained=True).to(self.device)
            self.eccv16_model.eval()
            logger.info("ECCV16 모델 로드 완료")
            
        except Exception as e:
            error_msg = f"모델 초기화 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
    
    def colorize_with_eccv16(self, image_path):
        """ECCV16 모델을 사용하여 흑백 이미지를 컬러로 변환합니다."""
        if not self.eccv16_model:
            raise ModelLoadError("ECCV16 모델이 로드되지 않았습니다")
        return self._process_image(image_path, self.eccv16_model)

    def _process_image(self, pil_data):
        model = self.eccv16_model
        """이미지를 로드하고 처리하여 컬러화된 이미지를 반환합니다."""
        try:
            # 이미지 로드 및 전처리
            logger.info(f"이미지 처리 중: {pil_data}")
            img = load_img_pil(pil_data)
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
            
            # 이미지 처리
            tens_l_rs = tens_l_rs.to(self.device)
            with torch.no_grad():
                out_ab = model(tens_l_rs)
            
            # 후처리 및 반환
            img_rgb = postprocess_tens(tens_l_orig, out_ab.cpu())
            out_img = Image.fromarray((img_rgb * 255).astype(np.uint8))
            logger.info("이미지 컬러화가 성공적으로 완료되었습니다.")
            return out_img
            
        except Exception as e:
            error_msg = f"이미지 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise    
        
    # def _process_image(self, image_path, model):
    #     """이미지를 로드하고 처리하여 컬러화된 이미지를 반환합니다."""
    #     try:
    #         if not os.path.exists(image_path):
    #             raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
    #         # 이미지 로드 및 전처리
    #         logger.info(f"이미지 처리 중: {image_path}")
    #         img = load_img(image_path)
    #         (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
            
    #         # 이미지 처리
    #         tens_l_rs = tens_l_rs.to(self.device)
    #         with torch.no_grad():
    #             out_ab = model(tens_l_rs)
            
    #         # 후처리 및 반환
    #         img_rgb = postprocess_tens(tens_l_orig, out_ab.cpu())
    #         out_img = Image.fromarray((img_rgb * 255).astype(np.uint8))
    #         logger.info("이미지 컬러화가 성공적으로 완료되었습니다.")
    #         return out_img
            
    #     except Exception as e:
    #         error_msg = f"이미지 처리 중 오류 발생: {str(e)}"
    #         logger.error(error_msg)
    #         raise
