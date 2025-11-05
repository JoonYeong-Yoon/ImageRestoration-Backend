import os, torch
from enum import Enum
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, Form
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from utils.exceptions import InvalidFileException, ModelNotLoadedException
from utils.auth import get_current_user
from utils.image import validate_image
from config.settings import UPLOAD_DIR, RESULT_DIR

from network.colorization_model import ColorizationModel
from network.colorization_model_unet import ColorizationUNetModel
from network.models import uformer

# ============================================================
# 공통 유틸
# ============================================================

def pad_to_divisible(x, div=16):
    _, _, h, w = x.size()
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    return F.pad(x, (0, pad_w, 0, pad_h)), h, w

class ProcessingMode(str, Enum):
    COLORIZE = "colorize"
    RESTORE = "restore"

router = APIRouter()

# ============================================================
# ✅ 전역 모델 캐싱 (로드 1회만 수행)
# ============================================================
print("[INFO] Initializing colorization models...")

try:
    UNET_MODEL = ColorizationUNetModel()
    ECCV16_MODEL = ColorizationModel()
    print("[INFO] ✅ Colorization models successfully loaded and cached.")
except Exception as e:
    print(f"[ERROR] ❌ Failed to initialize models: {e}")
    UNET_MODEL, ECCV16_MODEL = None, None

MODEL_DISPATCH = {
    "unet": lambda img: UNET_MODEL.colorize_with_unet(img) if UNET_MODEL else (_ for _ in ()).throw(ModelNotLoadedException("UNet 모델이 로드되지 않았습니다.")),
    "eccv16": lambda img: ECCV16_MODEL.colorize_with_eccv16(img) if ECCV16_MODEL else (_ for _ in ()).throw(ModelNotLoadedException("ECCV16 모델이 로드되지 않았습니다.")),
}

# ============================================================
# 🎨 /colorize : 흑백 → 컬러 복원
# ============================================================
@router.post("/colorize")
async def colorize(
    file: UploadFile = File(...),
    model: str = Form(..., enum=["unet", "eccv16", "UNET", "ECCV16"], description="사용할 모델 선택"),
):
    print("model",model)
    """흑백 이미지를 컬러로 변환 (UNet / ECCV16 선택 가능)"""
    validate_image(file)
    mode = ProcessingMode.COLORIZE
    user_id = "temp"

    safe_filename = f"{user_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    output_filename = f"{mode}d_{safe_filename}"
    output_path = os.path.join(RESULT_DIR, output_filename)

    try:
        # 1️⃣ 업로드 파일 저장
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # 2️⃣ PIL 로드
        pil_data = Image.open(input_path).convert("RGB")

        # 3️⃣ 선택한 모델 호출
        if model.lower() not in MODEL_DISPATCH:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 모델: {model}")

        print(f"[DEBUG] 모델 호출 시작: {model.lower()}, 입력 이미지 size: {pil_data.size}, mode: {pil_data.mode}")

        # =========================
        # 모델별 독립 _process_image 호출
        # =========================
        if model.lower() == "unet":
            print("unet")
            out_img = UNET_MODEL._process_image(pil_data)  # UNet 전용 처리
        elif model.lower() == "eccv16":
            print("eccv16")
            out_img = ECCV16_MODEL._process_image(pil_data)  # ECCV16 전용 처리

        print(f"[DEBUG] 모델 호출 완료: {model.lower()}, 출력 타입: {type(out_img)}, size: {out_img.size}")

        # 4️⃣ 결과 저장
        out_img.save(output_path)

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=f"colorized_{file.filename}"
        )

    except ValueError:
        raise ModelNotLoadedException()
    except Exception as e:
        import traceback
        print(f"[ERROR] {model} 처리 중 예외 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup 업로드 파일
        if os.path.exists(input_path):
            os.remove(input_path)
            

# ============================================================
# 전역 복원 모델 캐싱 (임시)
# ============================================================
print("[INFO] Initializing restoration models...")

try:
    # 아직 모델 구현 중이므로 임시 객체 생성
    UFORMER_MODEL = None  # 나중에 실제 Uformer 모델 로드 예정
    print("[INFO] ✅ Restoration model placeholder initialized.")
except Exception as e:
    print(f"[ERROR] ❌ Failed to initialize restoration model: {e}")
    UFORMER_MODEL = None

RESTORE_MODEL_DISPATCH = {
    "uformer": lambda img: (_ for _ in ()).throw(ModelNotLoadedException("Uformer 모델이 로드되지 않았습니다.")),
    # 나중에 다른 모델 추가 가능
}

# ============================================================
# 🛠 /restore : 훼손 이미지 복원 (임시 구조)
# ============================================================
@router.post("/restore")
async def restore(
    file: UploadFile = File(...),
    model: str = Form(..., enum=["uformer"], description="사용할 복원 모델 선택"),
):
    """훼손된 이미지를 복원"""
    validate_image(file)
    mode = ProcessingMode.RESTORE
    user_id = "temp"

    safe_filename = f"{user_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    output_filename = f"{mode}d_{safe_filename}"
    output_path = os.path.join(RESULT_DIR, output_filename)

    try:
        # 1️⃣ 업로드 파일 저장
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # 2️⃣ PIL 로드
        pil_data = Image.open(input_path).convert("RGB")

        # 3️⃣ 선택한 모델 호출
        if model.lower() not in RESTORE_MODEL_DISPATCH:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 복원 모델: {model}")

        print(f"[DEBUG] 복원 모델 호출 시작: {model.lower()}, 입력 이미지 size: {pil_data.size}, mode: {pil_data.mode}")

        # =========================
        # 실제 모델 구현 후 교체 예정
        # =========================
        out_img = RESTORE_MODEL_DISPATCH[model.lower()](pil_data)

        print(f"[DEBUG] 복원 모델 호출 완료: {model.lower()}, 출력 타입: {type(out_img)}, size: {out_img.size}")

        # 4️⃣ 결과 저장
        out_img.save(output_path)

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=f"restored_{file.filename}"
        )

    except ValueError:
        raise ModelNotLoadedException()
    except Exception as e:
        import traceback
        print(f"[ERROR] {model} 처리 중 예외 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup 업로드 파일
        if os.path.exists(input_path):
            os.remove(input_path)
