from models.user_model import get_user_by_email,insert_user
import bcrypt
import uuid
from utils.exceptions import EmailAlreadyExistsException, UserNotFoundException, InvalidCredentialsException

def register_user(email: str, password: str):
    existing = get_user_by_email(email)
    if existing:
        raise EmailAlreadyExistsException()
        
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
    uid = insert_user(email, hashed)
    print(uid)

    return {"user_id": uid}

def login_user( email: str, password: str):
    user = get_user_by_email(email)
    # user = db.query(User).filter(User.email == email).first()
    if not user:
        raise UserNotFoundException()
        
    # 저장된 비밀번호가 해시화되어 있는지 확인
    stored_password = user["password"]
    if stored_password.startswith(('$2a$', '$2b$', '$2y$')):  # bcrypt 해시 형식 확인
        if not bcrypt.checkpw(password.encode(), stored_password.encode()):
            raise InvalidCredentialsException()
    else:
        # 평문 비밀번호와 비교 (테스트용 계정)
        if password != stored_password:
            raise InvalidCredentialsException()
    
    # 세션 토큰 생성 및 DB 저장 (UUID4)
    session_token = uuid.uuid4().hex

    return {
        "access_token": session_token,
        "token_type": "session",
        "expires_in": 3600,
        "user_id": user["uid"]
    }

