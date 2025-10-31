
# 🖼️ Image Restoration VAE (Colorization)

PyTorch Lightning 기반 U-Net 구조를 이용한 흑백 → 컬러 이미지 복원 모델입니다.

## 📦 구조
models/               # 모델 정의 (UNet, Lightning)
utils/                # 전처리 및 색공간 변환
weights/              # 가중치 (state_dict)
train.py              # 학습 스크립트
inference.py          # 추론용 스크립트

## 🚀 사용법
```bash
pip install -r requirements.txt
python inference.py
