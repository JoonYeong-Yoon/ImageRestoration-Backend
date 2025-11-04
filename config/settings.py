import os
DB_CONFIG = {
    "user": "root",
    "password": "test",
    "host": "192.168.0.6",
    "port": 3307,
    "database": "ImageRestoration",
    "charset": "utf8mb4"
}

SECRET_KEY = "supersecretkey"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "exports/uploads")
RESULT_DIR = os.path.join(BASE_DIR, "exports/processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
