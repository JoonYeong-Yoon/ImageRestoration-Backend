from fastapi import Response
from fastapi import Request
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from services.auth_service import register_user, login_user
from schemas.auth import UserCreate, UserLogin, Token, ResponseModel

router = APIRouter()

@router.post("/register", response_model=ResponseModel)
async def register(user: UserCreate):
    result = register_user(user.email, user.password)
    return ResponseModel(ok=True, msg="회원가입 성공", data={"user_id": result["user_id"]})

@router.post("/login", response_model=Token)
async def login(user: UserLogin, response: Response):
    result = login_user(user.email, user.password)

    response = JSONResponse(
        content={
            "access_token": result["access_token"],
            "token_type": result["token_type"],
            "expires_in": result["expires_in"]
        }
    )
    # set session cookie (no Bearer prefix)
    response.set_cookie(
        key="session_token",
        value=result["access_token"],
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=result["expires_in"],
        path="/"
    )
    return response

@router.post("/logout", response_model=ResponseModel)
async def logout(request:Request):
    # read session token from cookie or Authorization header
    req: Request = request
    token = req.cookies.get("session_token") or (req.headers.get("Authorization") or "").split(" ")[-1]
    response = JSONResponse(content={"ok": True, "msg": "로그아웃 완료", "data": None})
    response.delete_cookie(key="session_token")
    return response
