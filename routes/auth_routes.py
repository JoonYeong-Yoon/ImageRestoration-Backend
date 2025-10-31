from flask import Blueprint, request, jsonify, session, make_response # make_response 추가
from controllers.auth_controller import register_user, login_user

auth_bp = Blueprint("auth_bp", __name__)

# ✅ 회원가입
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"ok": False, "msg": "이메일과 비밀번호를 입력하세요."}), 400

    result = register_user(email, password)
    return jsonify(result), (200 if result["ok"] else 409)


# ✅ 로그인 (수정된 부분)
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    print(data)
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"ok": False, "msg": "이메일과 비밀번호를 입력하세요."}), 400

    result = login_user(email, password)
    
    if result["ok"]:
        # 1. 응답 데이터 준비
        response_data = jsonify(result)
        # 2. make_response를 사용하여 응답 객체 생성
        response = make_response(response_data, 200)
        
        # 3. 명시적으로 쿠키 설정 (로그인 토큰 대신 user_id를 사용)
        # httponly=True: JS 접근 불가 (보안 강화)
        # samesite='Lax': 크로스 오리진 환경에서 쿠키 전송 허용
        # max_age=3600*24*7: 쿠키 유효기간 7일 설정
        response.set_cookie(
            'user_session', 
            str(result["user_id"]),  # user_id를 문자열로 변환하여 저장
            httponly=True, 
            samesite='Lax', 
            max_age=3600*24*7 
        )
        
        # NOTE: Flask의 내장 session은 이제 사용하지 않습니다.
        # session["user_id"] = result["user_id"] 
        
        return response
    else:
        return jsonify(result), 401


# ✅ 로그아웃 (수정된 부분)
@auth_bp.route("/logout", methods=["POST"])
def logout():
    # session.pop("user_id", None) # 내장 session 대신 쿠키 삭제 로직 추가
    
    response = make_response(jsonify({"ok": True, "msg": "로그아웃 완료!"}), 200)
    # user_session 쿠키를 삭제합니다 (만료 시간을 0으로 설정)
    response.set_cookie('user_session', '', expires=0, httponly=True, samesite='Lax')
    
    return response


# ✅ 로그인 상태 확인 (수정된 부분)
@auth_bp.route("/status", methods=["GET"])
def status():
    # 쿠키에서 'user_session' 값 (user_id)을 읽어옵니다.
    user_id = request.cookies.get("user_session") 
    
    if user_id:
        # NOTE: 이 user_id가 유효한지 DB에서 검증하는 로직이 추가되면 더 안전합니다.
        return jsonify({"logged_in": True, "user_id": user_id}), 200
    else:
        return jsonify({"logged_in": False}), 200
