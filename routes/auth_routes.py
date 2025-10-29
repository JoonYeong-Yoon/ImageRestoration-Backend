from flask import Blueprint
from controllers.auth_controller import register_controller, login_controller

auth_bp = Blueprint("auth", __name__)

auth_bp.route("/register", methods=["POST"])(register_controller)
auth_bp.route("/login", methods=["POST"])(login_controller)
