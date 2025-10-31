from flask import Blueprint, request, jsonify, session
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


# ✅ 로그인
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
        session["user_id"] = result["user_id"]
        return jsonify(result), 200
    else:
        return jsonify(result), 401


# ✅ 로그아웃
@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return jsonify({"ok": True, "msg": "로그아웃 완료!"}), 200


# ✅ 로그인 상태 확인
@auth_bp.route("/status", methods=["GET"])
def status():
    user_id = session.get("user_id")
    if user_id:
        return jsonify({"logged_in": True, "user_id": user_id}), 200
    else:
        return jsonify({"logged_in": False}), 200
