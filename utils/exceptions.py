from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

class CustomException(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)

async def exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "msg": exc.detail, "data": None}
    )

# Custom exceptions
class UserNotFoundException(CustomException):
    def __init__(self):
        super().__init__(404, "존재하지 않는 사용자입니다.")

class InvalidCredentialsException(CustomException):
    def __init__(self):
        super().__init__(401, "이메일 또는 비밀번호가 일치하지 않습니다.")

class EmailAlreadyExistsException(CustomException):
    def __init__(self):
        super().__init__(409, "이미 존재하는 이메일입니다.")

class InvalidFileException(CustomException):
    def __init__(self):
        super().__init__(400, "잘못된 파일 형식입니다.")

class ModelNotLoadedException(CustomException):
    def __init__(self):
        super().__init__(500, "AI 모델이 로드되지 않았습니다.")

class ImageProcessingException(CustomException):
    def __init__(self, detail: str):
        super().__init__(500, f"이미지 처리 중 오류가 발생했습니다: {detail}")