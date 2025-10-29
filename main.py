from flask import Flask
from flask_cors import CORS
from routes.auth_routes import auth_bp
from routes.image_routes import image_bp
from config.settings import SECRET_KEY
from db.connection import get_connection
from sqlalchemy import text

app = Flask(__name__)
CORS(app)
app.secret_key = SECRET_KEY

# ✅ DB 초기화

# ✅ 라우트 등록
app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(image_bp, url_prefix="/api/image")

# ✅ DB 연결 테스트용
@app.route("/test_db")
def test_db():
    try:
        db = get_connection()
        db.session.execute(text("SELECT 1"))
        return "✅ DB 연결 성공!"
    except Exception as e:
        return f"❌ DB 연결 실패: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
