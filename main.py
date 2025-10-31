# backend/main.py (최종 통합 및 모델 선택 버전)

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from routes.auth_routes import auth_bp
from routes.image_routes import image_bp
from config.settings import SECRET_KEY
from db.connection import get_connection
from sqlalchemy import text
import os, sys
import torch
from PIL import Image
import numpy as np 
import importlib.util # 모듈 임포트 시도용

# =================================================================
# 1. 경로 설정 및 AI 모델 로드 환경 설정
# =================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "ImageRestoration_Backend") 
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(PROJECT_DIR, "processed", "colorized_results") 
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 🌟 ImageRestoration_Backend를 Python path에 등록
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
    print("🔍 프로젝트 경로 등록 완료:", PROJECT_DIR)

# 🌟🌟🌟 [핵심 수정: 경로 재설정 및 함수 매핑] 🌟🌟🌟
# 임포트 오류 해결을 위해 colorization 폴더를 직접 sys.path에 추가하고 개별 파일을 임포트 시도
eccv16_func = None
siggraph17_func = None
load_colorizer_func = None
colorize_image_func = None
load_img = None
preprocess_img = None
postprocess_tens = None

try:
    COLORIZATION_DIR = os.path.join(PROJECT_DIR, "models", "colorization")
    MODEL_DIR = os.path.join(PROJECT_DIR, "models")
    
    # colorization 폴더와 models 폴더를 sys.path에 추가하여 파일을 모듈로 인식하도록 함
    if COLORIZATION_DIR not in sys.path:
        sys.path.insert(0, COLORIZATION_DIR)
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)

    # 📌 개별 파일 임포트
    import eccv16 
    import siggraph17
    import util
    import colorizer # colorizer.py는 models 폴더에 직접 있음
    
    # 임포트된 모듈에서 함수를 매핑합니다.
    eccv16_func = eccv16.eccv16
    siggraph17_func = siggraph17.siggraph17
    load_colorizer_func = colorizer.load_colorizer
    colorize_image_func = colorizer.colorize_image
    load_img = util.load_img
    preprocess_img = util.preprocess_img
    postprocess_tens = util.postprocess_tens
    
    print("✅ AI 모델 파일 임포트 성공 (경로 재설정)")
except Exception as e:
    print(f"❌ 최종 AI 모델 파일 임포트 실패: {e}")
    print("-> models/colorization 폴더 구조와 colorizer.py 파일의 함수 정의를 확인하세요.")


# =================================================================
# 2. Flask 앱 초기화
# =================================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.secret_key = SECRET_KEY


# =================================================================
# 3. 모델 로드 (서버 시작 시 한 번만 로드)
# =================================================================

# 💡 팀 모델 로드
team_model = None
try:
    if load_colorizer_func:
        MODEL_PATH = os.path.join(PROJECT_DIR, "models", "colorizer.ckpt") # colorizer.ckpt는 models 폴더에 있음
        team_model = load_colorizer_func(MODEL_PATH)
        print(f"✅ 팀 모델 로드 완료: {MODEL_PATH}")
    else:
        raise ImportError("load_colorizer_func is not defined.")
except Exception as e:
    print(f"❌ 팀 모델 로드 실패: {e}")

# 💡 ECCV16 모델 로드
eccv_model = None
try:
    if eccv16_func:
        eccv_model = eccv16_func(pretrained=True).eval() 
        print(f"✅ ECCV16 모델 로드 완료 (가중치 자동 다운로드)")
except Exception as e:
    print(f"❌ ECCV16 모델 로드 실패: {e}")

# 💡 SIGGRAPH17 모델 로드
siggraph_model = None
try:
    if siggraph17_func:
        siggraph_model = siggraph17_func(pretrained=True).eval() 
        print(f"✅ SIGGRAPH17 모델 로드 완료 (가중치 자동 다운로드)")
except Exception as e:
    print(f"❌ SIGGRAPH17 모델 로드 실패: {e}")


# =================================================================
# 4. 라우트 등록 및 5. AI 컬러화 API
# =================================================================

app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(image_bp, url_prefix="/api/image")

@app.route("/test_db")
def test_db():
    try:
        db = get_connection()
        db.session.execute(text("SELECT 1"))
        return "✅ DB 연결 성공!"
    except Exception as e:
        return f"❌ DB 연결 실패: {e}"

# 💡 모델 호출 및 실행을 위한 공통 함수
def run_colorizer_dispatch(model_type, uploaded_file):
    
    # 1. 파일 저장
    filename = uploaded_file.filename
    input_path = os.path.join(UPLOAD_DIR, filename)
    uploaded_file.save(input_path)
    
    result_path = ""
    
    if model_type == 'team':
        # 팀 모델 로직: colorize_image_func 사용 (파일 경로와 결과 디렉토리를 인수로 받음)
        if team_model is None:
            raise Exception("팀 모델이 로드되지 않았습니다.")
        result_path = colorize_image_func(team_model, input_path, RESULT_DIR)
        
    elif model_type in ['eccv16', 'siggraph17']:
        # ECCV/SIGGRAPH 모델 로직: PyTorch 모델 객체와 util 함수 사용
        model = eccv_model if model_type == 'eccv16' else siggraph_model
        
        if model is None:
            raise Exception(f"{model_type.upper()} 모델이 로드되지 않았습니다.")

        img = load_img(input_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        
        with torch.no_grad():
            output = model(tens_l_rs).cpu()
        out_img = postprocess_tens(tens_l_orig, output)

        output_filename = f"{model_type}_{os.path.basename(filename)}"
        result_path = os.path.join(RESULT_DIR, output_filename)
        Image.fromarray((out_img * 255).astype(np.uint8)).save(result_path)
        
    else:
        raise ValueError("Invalid model_type specified.")
    
    # 2. 임시 입력 파일 삭제 (선택 사항)
    os.remove(input_path)
    
    return result_path


# 🌟🌟🌟 [통합 API 엔드포인트] 🌟🌟🌟
@app.route("/api/ai/colorize", methods=["POST"])
def ai_colorize_unified():
    """
    하나의 엔드포인트에서 model_type 파라미터에 따라 다른 모델을 실행합니다.
    Postman 요청 시 'model_type' 필드를 추가해야 합니다.
    """
    model_type = None
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "msg": "파일이 포함되지 않았습니다."}), 400
        
        # Postman의 Body -> form-data에서 model_type 필드를 읽어옴
        model_type = request.form.get("model_type") 
        
        if not model_type or model_type not in ['team', 'eccv16', 'siggraph17']:
            return jsonify({"ok": False, "msg": "모델 타입(model_type)을 'team', 'eccv16', 'siggraph17' 중 하나로 form-data에 지정해주세요."}), 400
        
        print(f"🎨 {model_type.upper()} 모델로 컬러화 시작")
        
        result_path = run_colorizer_dispatch(model_type, request.files["file"])
        
        print(f"🎉 컬러화 완료: {result_path}")

        return send_file(result_path, mimetype="image/png")

    except Exception as e:
        print(f"❌ 컬러화 오류 ({model_type}):", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

# =================================================================
# 6. 앱 실행
# =================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)