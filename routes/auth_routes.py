from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from db.connection import get_db
from services.auth_service import register_user, login_user, logout_user
from schemas.auth import UserCreate, UserLogin, Token, ResponseModel
from utils.auth import get_current_user
from utils.exceptions import InvalidCredentialsException

router = APIRouter()

@router.post("/register", response_model=ResponseModel)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    result = register_user(db, user.email, user.password)
    return ResponseModel(ok=True, msg="회원가입 성공", data={"user_id": result["user_id"]})

@router.post("/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    result = login_user(db, user.email, user.password)

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
async def logout(request, db: Session = Depends(get_db)):
    # read session token from cookie or Authorization header
    from fastapi import Request
    req: Request = request
    token = req.cookies.get("session_token") or (req.headers.get("Authorization") or "").split(" ")[-1]
    if token:
        try:
            logout_user(db, token)
        except Exception:
            pass
    response = JSONResponse(content={"ok": True, "msg": "로그아웃 완료", "data": None})
    response.delete_cookie(key="session_token")
    return response

@router.get("/me", response_model=ResponseModel)
async def get_user_info(current_user: dict = Depends(get_current_user)):
    return ResponseModel(
        ok=True,
        msg="사용자 정보 조회 성공",
        data={"user_id": current_user["user_id"], "email": current_user["email"]}
    )
