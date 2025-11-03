import os
import torch
from PIL import Image
import torchvision.transforms as T
from typing import Literal

class ImageRestorationService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Colorization 모델 로드
        try:
            from modeling.colorization_model import ColorizationModel
            self.colorization_model = ColorizationModel()
            print(f"컬러화 모델이 성공적으로 로드되었습니다. 사용 중인 디바이스: {self.device}")
        except Exception as e:
            print(f"컬러화 모델 로드 실패: {str(e)}")
            self.colorization_model = None
        
        # Damage Restoration 모델 로드
        try:
            self.restoration_model = None
            self.restoration_model_path = os.path.join("damagedRestoration", "Uformer_B.pth")
            if os.path.exists(self.restoration_model_path):
                from modeling.restoration_model import RestorationModel  # 복원 모델 클래스 import
                self.restoration_model = RestorationModel(self.restoration_model_path)
                print(f"복원 모델이 성공적으로 로드되었습니다. 사용 중인 디바이스: {self.device}")
            else:
                print(f"복원 모델 체크포인트를 찾을 수 없습니다: {self.restoration_model_path}")
        except Exception as e:
            print(f"복원 모델 로드 실패: {str(e)}")
            self.restoration_model = None
        
        # 변환기 설정
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def process_image(self, 
                     input_path: str, 
                     output_path: str, 
                     mode: Literal["colorize", "restore"]) -> str:
        """
        이미지 처리 함수
        Args:
            input_path: 입력 이미지 경로
            output_path: 출력 이미지 저장 경로
            mode: "colorize" 또는 "restore"
        Returns:
            처리된 이미지 경로
        """
        try:
            # 이미지 로드 및 전처리
            img = Image.open(input_path)
            if mode == "colorize":
                img = img.convert('L')  # 흑백 변환
            
            # 원본 크기 저장
            original_size = img.size
            
            # 텐서 변환
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            if mode == "colorize":
                if self.colorization_model is None:
                    raise ValueError("컬러화 모델이 로드되지 않았습니다.")
                try:
                    output = self.colorization_model.colorize_with_eccv16(input_path)
                except Exception as e:
                    raise ValueError(f"컬러화 처리 실패: {str(e)}")
            else:  # restore
                if self.restoration_model is None:
                    raise ValueError("복원 모델이 로드되지 않았습니다.")
                output = self.restoration_model.process_image(input_path)
                
            # 원본 크기로 리사이즈
            output = output.resize(original_size, Image.BICUBIC)
            
            # 저장
            output.save(output_path)
            return output_path
                
        except Exception as e:
            raise Exception(f"Image processing failed: {str(e)}")

# 싱글톤 인스턴스 생성
restoration_service = ImageRestorationService()