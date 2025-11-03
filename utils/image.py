import os
from fastapi import UploadFile, HTTPException

from utils.exceptions import InvalidFileException, ModelNotLoadedException

ALLOWED_EXTENSIONS = {"jpeg", "jpg", "png", "bmp"}



def validate_image(file):
    # Check file extension
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileException()
    
    # Check file content type
    content_type = file.content_type
    if not content_type.startswith("image/"):
        raise InvalidFileException()


# async def process_image(file: UploadFile, mode: ProcessingMode, user_id: int) -> str:
#     validate_image(file)
    
#     # Create unique filenames
#     safe_filename = f"{user_id}_{file.filename}"
#     input_path = os.path.join(UPLOAD_DIR, safe_filename)
#     output_filename = f"{mode}d_{safe_filename}"
#     output_path = os.path.join(RESULT_DIR, output_filename)
    
#     try:
#         # Save uploaded file
#         content = await file.read()
#         with open(input_path, "wb") as f:
#             f.write(content)
            
#         # Process image
#         processed_path = restoration_service.process_image(
#             input_path=input_path,
#             output_path=output_path,
#             mode=mode
#         )
        
#         return processed_path
#     except ValueError as e:
#         raise ModelNotLoadedException()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         # Cleanup uploaded file
#         if os.path.exists(input_path):
#             os.remove(input_path)