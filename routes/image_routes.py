from flask import Blueprint, request, jsonify, send_file
from services.enhancer_service import colorize_image
import os

image_bp = Blueprint("image_bp", __name__)

@image_bp.route("/colorize", methods=["POST"])
def colorize():
    """흑백 이미지 → AI 컬러화"""
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "이미지 파일이 필요합니다."}), 400

    try:
        # AI 모델로 컬러화 처리
        output_path = colorize_image(file)

        if not os.path.exists(output_path):
            return jsonify({"error": "결과 파일이 생성되지 않았습니다."}), 500

        print(f"✅ 컬러화 완료 → {output_path}")
        # 결과 이미지를 바로 응답으로 반환
        return send_file(output_path, mimetype="image/png")

    except FileNotFoundError as e:
        # 모델 파일이 없을 때
        return jsonify({"error": f"모델 파일을 찾을 수 없습니다: {str(e)}"}), 500
    except Exception as e:
        # 기타 예외 처리
        return jsonify({"error": str(e)}), 500
