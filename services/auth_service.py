from db.connection import get_connection
import bcrypt

def register_user(email, password):
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cur.fetchone():
            return {"ok": False, "msg": "이미 존재하는 이메일입니다."}

        cur.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_pw))
        return {"ok": True, "msg": "회원가입 성공"}

def login_user(email, password):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()

        if not user:
            return {"ok": False, "msg": "존재하지 않는 이메일입니다."}

        if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
            return {"ok": False, "msg": "비밀번호가 일치하지 않습니다."}

        return {"ok": True, "msg": "로그인 성공", "user": {"id": user["id"], "email": user["email"]}}
