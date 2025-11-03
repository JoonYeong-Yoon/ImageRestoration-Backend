import os, torch
from enum import Enum
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from utils.exceptions import InvalidFileException, ModelNotLoadedException

from utils.auth import get_current_user
from utils.image import validate_image
from config.settings import UPLOAD_DIR,RESULT_DIR
from network.restoration_model import RestorationModel
from network.colorization_model import ColorizationModel
from network.models import uformer
# from services.restoration_service import restoration_service

def pad_to_divisible(x, div=16):
    _, _, h, w = x.size()
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    return F.pad(x, (0, pad_w, 0, pad_h)), h, w

class ProcessingMode(str, Enum):
    COLORIZE = "colorize"
    RESTORE = "restore"

router = APIRouter()

@router.post("/colorize")
async def colorize(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):

    """흑백 이미지를 컬러로 변환"""
    validate_image(file)
    mode = ProcessingMode.COLORIZE
    user_id = current_user["user_id"]
    safe_filename = f"{user_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    output_filename = f"{mode}d_{safe_filename}"
    output_path = os.path.join(RESULT_DIR, output_filename)
    
    # Save uploaded file
    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        # todo - > RESIZE 및 모델로드 부분 분리
        colorize_model = ColorizationModel()
        pil_data = Image.open(input_path)
        out_img = colorize_model._process_image(pil_data)
        # output = output.resize(original_size, Image.BICUBIC)
        out_img.save(output_path)

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=f"colorized_{file.filename}"
        )
    except ValueError as e:
        raise ModelNotLoadedException()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if os.path.exists(input_path):
            os.remove(input_path)
            


@router.post("/restore")
async def restore(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """훼손된 이미지 복원"""
    """흑백 이미지를 컬러로 변환"""
    validate_image(file)
    mode = ProcessingMode.COLORIZE
    user_id = current_user["user_id"]
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
            