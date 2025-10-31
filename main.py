from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from routes.auth_routes import auth_bp
from routes.image_routes import image_bp
from config.settings import SECRET_KEY
from db.connection import get_connection
from sqlalchemy import text
import os, sys

# =================================================================
# 1. 경로 설정 및 AI 모델 로드
# =================================================================

# ✅ 백엔드 절대 경로 인식 (ImageRestoration_Backend 폴더 등록)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 💡 수정된 경로: 'backend' 폴더를 명시적으로 추가하여 colorizer 모듈을 찾을 수 있도록 함
MODEL_DIR = os.path.join(BASE_DIR, "backend", "ImageRestoration_Backend") 

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
    print("🔍 sys.path 등록 완료:", MODEL_DIR)

# ✅ AI 컬러화 모델 import
# (MODEL_DIR이 sys.path에 추가되었기 때문에 이제 colorizer 모듈을 찾을 수 있음)
from colorizer import load_colorizer, colorize_image 

# ✅ Flask 앱 초기화
app = Flask(__name__)
# CORS 설정: 모든 도메인에서의 요청을 허용 (개발 환경용)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.secret_key = SECRET_KEY

# ✅ 디렉토리 설정
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ✅ 팀장님 모델 로드
# 모델 로드는 서버 시작 시 한 번만 수행
try:
    MODEL_PATH = os.path.join(MODEL_DIR, "colorizer.ckpt")
    model = load_colorizer(MODEL_PATH)
    print(f"✅ AI 모델 로드 완료: {MODEL_PATH}")
except Exception as e:
    print(f"❌ AI 모델 로드 실패: {e}")
    # 모델 로드 실패 시 앱 실행을 막지 않기 위해 예외 처리

# =================================================================
# 2. 라우트 및 API 정의
# =================================================================

# ✅ 라우트 등록 (auth_bp, image_bp는 외부 파일에 정의됨)
app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(image_bp, url_prefix="/api/image")

# ✅ DB 연결 테스트용
@app.route("/test_db")
def test_db():
    try:
        db = get_connection()
        # 간단한 쿼리 실행
        db.session.execute(text("SELECT 1")) 
        return "✅ DB 연결 성공!"
    except Exception as e:
        return f"❌ DB 연결 실패: {e}"

# ✅ AI 컬러화 API (흑백 → 컬러)
@app.route("/api/ai/colorize", methods=["POST"])
def ai_colorize():
    try:
        # 1. 파일 존재 여부 확인 (키 이름은 'file' 임)
        if "file" not in request.files:
            return jsonify({"ok": False, "msg": "요청에 'file' 키로 파일이 포함되어 있지 않습니다."}), 400

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"ok": False, "msg": "파일 이름이 없습니다."}), 400

        filename = file.filename

        # 2. 파일 저장
        input_path = os.path.join(UPLOAD_DIR, filename)
        file.save(input_path)

        # 3. AI 모델 처리 (모델이 성공적으로 로드되었다고 가정)
        if 'model' in globals() and model is not None:
            print(f"🎨 AI 컬러화 시작: {filename}")
            result_path = colorize_image(model, input_path, RESULT_DIR)
            print(f"🎉 AI 컬러화 완료 및 저장: {result_path}")

            # 4. 처리된 이미지 파일 응답
            # Postman의 Response 탭에서 바로 이미지 확인 가능
            return send_file(result_path, mimetype="image/png")
        else:
            return jsonify({"ok": False, "msg": "AI 모델이 초기화되지 않았습니다."}), 503

    except Exception as e:
        print("❌ AI 컬러화 오류:", e)
        # 디버깅을 위해 상세 오류 메시지 제공
        return jsonify({"ok": False, "msg": f"서버 처리 오류 발생: {str(e)}"}), 500


# =================================================================
# 3. 앱 실행
# =================================================================
if __name__ == "__main__":
    # host를 "0.0.0.0"으로 설정하여 외부 IP(192.168.0.51)에서도 접근 가능하게 함
    app.run(host="0.0.0.0", port=5000, debug=True)