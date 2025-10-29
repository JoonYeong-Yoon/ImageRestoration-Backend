from flask import Flask
from flask_cors import CORS
from routes.auth_routes import auth_bp
# from routes.aa_routes import auth_aa
from models.user_model import create_user_table
from config.settings import SECRET_KEY


app = Flask(__name__)
CORS(app)
app.secret_key = SECRET_KEY

# 라우트 등록
app.register_blueprint(auth_bp, url_prefix="/api/auth")
# app.register_blueprint(auth_aa, url_prefix="/api/aa")

# 서버 시작 전 DB 테이블 자동 생성
create_user_table()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
