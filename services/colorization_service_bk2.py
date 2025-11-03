# services/colorization_service.py
import os
from PIL import Image
from network.colorization_model import ColorizationModel

class ColorizationService:
    def __init__(self):
        try:
            self.model = ColorizationModel()
        except Exception as e:
            print(f"컬러화 모델 로드 실패: {str(e)}")
            self.model = None

    def colorize_image(self, input_path: str, output_filename: str = None) -> str:
        if self.model is None:
            raise RuntimeError("컬러화 모델이 로드되지 않았습니다.")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

        # 컬러화 수행
        out_img = self.model.colorize(input_path)

        # 원본 크기로 리사이즈
        orig_img = Image.open(input_path)
        out_img = out_img.resize(orig_img.size, Image.BICUBIC)

        # processed 폴더 절대 경로
        processed_dir = os.path.join(os.getcwd(), "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # 출력 파일명 설정
        if output_filename is None:
            output_filename = "colorized_" + os.path.basename(input_path)
        output_path = os.path.join(processed_dir, output_filename)

        # 저장 (PIL 사용)
        out_img.save(output_path)

        # 브라우저에서 접근할 수 있는 URL 반환
        url_path = f"/processed/{output_filename}"
        return url_path


# 싱글톤 인스턴스
colorization_service = ColorizationService()
