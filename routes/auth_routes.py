from flask import Blueprint, request, jsonify, session, make_response 
from controllers.auth_controller import register_user, login_user

auth_bp = Blueprint("auth_bp", __name__)

# ✅ 회원가입
@auth_bp.route("/register", methods=["POST"])
def register():
    """새로운 사용자를 등록 처리합니다."""
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"ok": False, "msg": "이메일과 비밀번호를 입력하세요."}), 400

    result = register_user(email, password)
    return jsonify(result), (200 if result["ok"] else 409)


# ✅ 로그인 (최종 확정 버전: HTTP 개발 환경에 최적화)
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"ok": False, "msg": "이메일과 비밀번호를 입력하세요."}), 400

    result = login_user(email, password)
    
    if result["ok"]:
        response_data = jsonify(result)
        response = make_response(response_data, 200)
        
        # 🚨 HTTP 환경에서 가장 안정적인 설정입니다.
        # SameSite='Lax'와 secure=False를 사용하여 크로스 오리진/HTTP 환경에서 쿠키 유지
        response.set_cookie(
            'user_session', 
            str(result["user_id"]),  
            httponly=True, 
            samesite='Lax',      # <--- 이 부분이 'Lax' 설정입니다.
            secure=False,        # Secure 플래그 제거 (HTTP 필수)
            max_age=3600*24*7,   # 7일 유지 (영구 쿠키)
            path='/'             
        )
        
        return response
    else:
        return jsonify(result), 401


# ✅ 로그아웃 (HTTP 환경 설정에 맞게 롤백)
@auth_bp.route("/logout", methods=["POST"])
def logout():
    """쿠키를 만료시켜 로그아웃 처리합니다."""
    response = make_response(jsonify({"ok": True, "msg": "로그아웃 완료!"}), 200)
    # 로그아웃 시에도 Lax 정책을 적용
    response.set_cookie('user_session', '', expires=0, httponly=True, samesite='Lax', secure=False, path='/') 
    
    return response


# ✅ 로그인 상태 확인 (동일)
@auth_bp.route("/status", methods=["GET"])
def status():
    """쿠키를 확인하여 현재 로그인 상태를 반환합니다."""
    user_id = request.cookies.get("user_session") 
    
    if user_id:
        return jsonify({"logged_in": True, "user_id": user_id}), 200
    else:
        return jsonify({"logged_in": False}), 200
