from fastapi import Request, HTTPException, Depends
from sqlalchemy.orm import Session
# from models.user_model import User
from datetime import datetime
from models.user_model import get_user_by_email

async def get_current_user(request: Request):
    """Retrieve current user from session token stored in cookie or Bearer header.

    Looks for cookie 'session_token' first, then Authorization: Bearer <token>.
    Validates the token against the `sessions` table and returns a dict with
    `user_id` and `email` on success. Raises 401 on failure.
    """
    token = request.cookies.get("session_token")
    # if not token:
    #     auth = request.headers.get("Authorization")
    #     if auth and auth.startswith("Bearer "):
    #         token = auth.split(" ", 1)[1]

    # if not token:
    #     raise HTTPException(status_code=401, detail="Authentication credentials were not provided")

    # sess = db.query(SessionToken).filter(SessionToken.session_id == token).first()
    # if not sess:
    #     raise HTTPException(status_code=401, detail="Invalid or expired session token")

    # if sess.expires_at and sess.expires_at < datetime.utcnow():
    #     # session expired
    #     # optionally delete expired session
    #     try:
    #         db.delete(sess)
    #         db.commit()
    #     except Exception:
    #         pass
    #     raise HTTPException(status_code=401, detail="Session expired")
    # user = get_user_by_email(email)
    # user = db.query(User).filter(User.uid == sess.uid).first()
    # if not user:
    #     raise HTTPException(status_code=401, detail="User not found for session")

    # return {"user_id": user.uid, "email": user.email}