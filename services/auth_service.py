from sqlalchemy.orm import Session
from models.user_model import User
from db_models.session_model import SessionToken
import bcrypt
import uuid
from datetime import datetime, timedelta

def register_user(db: Session, email: str, password: str):
    from utils.exceptions import EmailAlreadyExistsException
    
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise EmailAlreadyExistsException()
        
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
    user = User(email=email, password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"user_id": user.uid}

def login_user(db: Session, email: str, password: str):
    from utils.exceptions import UserNotFoundException, InvalidCredentialsException
    
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise UserNotFoundException()
        
    # 저장된 비밀번호가 해시화되어 있는지 확인
    stored_password = user.password
    if stored_password.startswith(('$2a$', '$2b$', '$2y$')):  # bcrypt 해시 형식 확인
        if not bcrypt.checkpw(password.encode(), stored_password.encode()):
            raise InvalidCredentialsException()
    else:
        # 평문 비밀번호와 비교 (테스트용 계정)
        if password != stored_password:
            raise InvalidCredentialsException()
    
    # 세션 토큰 생성 및 DB 저장 (UUID4)
    session_token = uuid.uuid4().hex
    expires_at = datetime.utcnow() + timedelta(hours=1)

    session_entry = SessionToken(
        session_id=session_token,
        uid=user.uid,
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )
    db.add(session_entry)
    db.commit()

    return {
        "access_token": session_token,
        "token_type": "session",
        "expires_in": 3600,
        "user_id": user.uid
    }

def logout_user(db: Session, session_token: str):
    """Delete session token from DB to log out user."""
    sess = db.query(SessionToken).filter(SessionToken.session_id == session_token).first()
    if sess:
        db.delete(sess)
        db.commit()
        return True
    return False
