from sqlalchemy import text
from db.connection import engine

def get_user_by_email(email):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM users WHERE email = :email LIMIT 1"),
            {"email": email}  # 파라미터 바인딩 (SQL 인젝션 방지)
        )
        row = result.fetchone()
        if row:
            return dict(row._mapping)  # Row 객체 → dict 변환
        return None
    
def insert_user(email, hashed):
    with engine.connect() as conn:
        result = conn.execute(
            text("INSERT INTO users (email, password) VALUES (:email, :password)"),
            {"email": email, "password": hashed}
        )
        conn.commit() 
        uid = result.lastrowid  
        return uid
