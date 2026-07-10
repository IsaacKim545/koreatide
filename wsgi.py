"""WSGI 진입점 (gunicorn/Render용).

    gunicorn wsgi:app --bind 0.0.0.0:$PORT

서비스키는 환경변수 KHOA_API_KEY 로 주입합니다(파일 커밋 금지).
"""
from web.app import app  # noqa: F401

if __name__ == "__main__":
    app.run()
