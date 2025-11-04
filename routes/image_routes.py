import os, torch
from enum import Enum
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
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
    model: str = Query("eccv16", enum=["unet", "eccv16"], description="사용할 모델 선택"),
):
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
            out_img = UNET_MODEL._process_image(pil_data)  # UNet 전용 처리
        elif model.lower() == "eccv16":
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
            

@router.post("/restore")
async def restore(
    file: UploadFile = File(...),
    # current_user: dict = Depends(get_current_user)
):
    """훼손된 이미지 복원"""
    """흑백 이미지를 컬러로 변환"""
    validate_image(file)
    mode = ProcessingMode.COLORIZE
    # user_id = current_user["user_id"]
    user_id = "temp"
    safe_filename = f"{user_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    output_filename = f"{mode}d_{safe_filename}"
    output_path = os.path.join(RESULT_DIR, output_filename)
    # Save uploaded file
    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        restoration_model = uformer.UNet(dim = 32)
        weight_file_path = "network/weights/damageRestoration/Uformer_B.pth"
        
        checkpoint = torch.load(weight_file_path, map_location="cpu")

        # checkpoint가 dict 구조인지 확인
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model_dict = restoration_model.state_dict()
        # 맞는 키만 업데이트
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        restoration_model.load_state_dict(model_dict)
        restoration_model.eval()

        restoration_weights = torch.load(weight_file_path,map_location="cpu")
        restoration_model.load_state_dict(restoration_weights)
        restoration_model.eval()
        # todo - > RESIZE 및 모델로드 부분 분리
        transform = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        img = Image.open(input_path).convert("RGB")
        orig_w, orig_h = img.size

        img_tensor = transform(img).unsqueeze(0)
        padded_tensor, orig_h, orig_w = pad_to_divisible(img_tensor, div=16)
        with torch.no_grad():
            output_tensor = restoration_model(padded_tensor)

        # crop 원래 크기로
        output_tensor = output_tensor[:, :, :orig_h, :orig_w]

        # ====== 후처리 및 저장 ======
        output_img = output_tensor.squeeze(0).cpu()
        output_img = to_pil(output_img.clamp(0, 1))
        output_img.save(output_path)

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=f"restored_{file.filename}"
        )

    except ValueError as e:
        raise ModelNotLoadedException()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if os.path.exists(input_path):
            os.remove(input_path)
            