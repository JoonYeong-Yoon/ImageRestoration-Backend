from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from utils.exceptions import InvalidFileException, ModelNotLoadedException
from remove.utils.auth import get_current_user
from backend.services.restoration_service_bk import restoration_service
import os
from typing import List
from enum import Enum

router = APIRouter()
UPLOAD_DIR = "uploads"
RESULT_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpeg", "jpg", "png", "bmp"}

class ProcessingMode(str, Enum):
    COLORIZE = "colorize"
    RESTORE = "restore"

def validate_image(file: UploadFile):
    # Check file extension
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileException()
    
    # Check file content type
    content_type = file.content_type
    if not content_type.startswith("image/"):
        raise InvalidFileException()

async def process_image(file: UploadFile, mode: ProcessingMode, user_id: int) -> str:
    validate_image(file)
    
    # Create unique filenames
    safe_filename = f"{user_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    output_filename = f"{mode}d_{safe_filename}"
    output_path = os.path.join(RESULT_DIR, output_filename)
    
    try:
        # Save uploaded file
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
            
        # Process image
        processed_path = restoration_service.process_image(
            input_path=input_path,
            output_path=output_path,
            mode=mode
        )
        
        return processed_path
    except ValueError as e:
        raise ModelNotLoadedException()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if os.path.exists(input_path):
            os.remove(input_path)

@router.post("/colorize")
async def colorize(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """흑백 이미지를 컬러로 변환"""
    output_path = await process_image(
        file=file,
        mode=ProcessingMode.COLORIZE,
        user_id=current_user["user_id"]
    )
    
    return FileResponse(
        output_path,
        media_type="image/png",
        filename=f"colorized_{file.filename}"
    )

@router.post("/restore")
async def restore(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """훼손된 이미지 복원"""
    output_path = await process_image(
        file=file,
        mode=ProcessingMode.RESTORE,
        user_id=current_user["user_id"]
    )
    
    return FileResponse(
        output_path,
        media_type="image/png",
        filename=f"restored_{file.filename}"
    )
