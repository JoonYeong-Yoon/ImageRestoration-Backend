from db.connection import execute_query
from werkzeug.security import generate_password_hash, check_password_hash

# ✅ 회원가입
def create_user(email, password):
    """새로운 사용자 추가"""
    # 중복 확인
    check_sql = "SELECT * FROM users WHERE email = %s"
    existing_user = execute_query(check_sql, (email,), fetchone=True)
    if existing_user:
        return {"ok": False, "msg": "이미 존재하는 이메일입니다."}

    hashed_pw = generate_password_hash(password)
    insert_sql = "INSERT INTO users (email, password) VALUES (%s, %s)"
    execute_query(insert_sql, (email, hashed_pw))
    return {"ok": True, "msg": "회원가입 성공!"}


# ✅ 로그인 검증
def validate_user(email, password):
    """사용자 로그인 검증"""
    sql = "SELECT * FROM users WHERE email = %s"
    user = execute_query(sql, (email,), fetchone=True)
    print(user)
    if not user:
        return None
    if not check_password_hash(user["password"], password):
        return None
    return user
