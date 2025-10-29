from services.auth_service import register_user, login_user
from flask import request, jsonify

def register_controller():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"ok": False, "msg": "이메일과 비밀번호를 입력하세요."}), 400

    result = register_user(email, password)
    return jsonify(result), (200 if result["ok"] else 400)

def login_controller():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"ok": False, "msg": "이메일과 비밀번호를 입력하세요."}), 400

    result = login_user(email, password)
    return jsonify(result), (200 if result["ok"] else 401)
