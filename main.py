import os, sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.auth_routes import router as auth_router
from routes.image_routes import router as image_router
from utils.exceptions import CustomException, exception_handler
app = FastAPI(
    title="AI Image Restoration API",
    version="1.0",
    description="AI 이미지 복원 및 사용자 인증 API",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프로덕션에서는 실제 도메인으로 변경
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],)

# =====================
# 2. 업로드/결과 디렉토리
# =====================

# =====================
# 3. 라우터 등록
# =====================
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(image_router, prefix="/api/images", tags=["Image"])


# # Ensure DB tables for new models exist on startup (creates `sessions` table if missing)
# @app.on_event("startup")
# def startup_event():
#     try:
#         from db.connection import engine, Base
#         # import model modules so their metadata is registered
#         import db_models.user_model  # noqa: F401
#         import db_models.session_model  # noqa: F401
#         Base.metadata.create_all(bind=engine)
#     except Exception as e:
#         # don't crash the app on startup table creation failure; log and continue
#         print(f"Warning: failed to create DB tables on startup: {e}")

# Exception handlers
app.add_exception_handler(CustomException, exception_handler)

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        sys.exit(1)
