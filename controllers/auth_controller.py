from models.user_model import create_user, validate_user

# ✅ 회원가입
def register_user(email, password):
    result = create_user(email, password)
    return result


# ✅ 로그인
def login_user(email, password):
    user = validate_user(email, password)
    if not user:
        return {"ok": False, "msg": "이메일 또는 비밀번호가 잘못되었습니다."}
    return {"ok": True, "msg": "로그인 성공!", "user_id": user["uid"]}
