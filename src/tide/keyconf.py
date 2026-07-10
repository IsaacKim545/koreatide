"""KHOA 인증키/엔드포인트 설정 관리.

키 우선순위: 명시적 인자 > 환경변수 KHOA_API_KEY > 설정파일(config/khoa_key.txt).
한 번 저장(save_service_key)해두면 이후 --sample 없이 전국 조회가 됩니다.

엔드포인트: 기본은 KHOA 바다누리(oceandata) 엔드포인트. 공공데이터포털(data.go.kr)에서
키를 받은 경우 base_url이 다를 수 있어 KHOA_API_URL 환경변수나 --api-url로 바꿉니다.

주의: khoa_key.txt 는 개인 인증키이므로 공유/커밋하지 마세요(.gitignore에 포함).
"""
from __future__ import annotations

import os
from typing import Optional

# 공공데이터포털(apis.data.go.kr) 인증 엔드포인트 (KHOA 조위관측소 실측·예측 조위)
DEFAULT_API_URL = "https://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService"


def _key_file() -> str:
    here = os.path.dirname(os.path.abspath(__file__))          # src/tide
    return os.path.join(here, "..", "..", "config", "khoa_key.txt")


def load_service_key(cli_key: Optional[str] = None) -> Optional[str]:
    """우선순위에 따라 서비스키를 반환. 없으면 None."""
    if cli_key:
        return cli_key.strip()
    env = os.environ.get("KHOA_API_KEY")
    if env:
        return env.strip()
    path = _key_file()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                k = f.read().strip()
            return k or None
        except Exception:  # noqa
            return None
    return None


def save_service_key(key: str) -> str:
    """서비스키를 config/khoa_key.txt 에 저장. 저장 경로 반환."""
    path = _key_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(key.strip())
    return path


def get_api_url(cli_url: Optional[str] = None) -> str:
    return (cli_url or os.environ.get("KHOA_API_URL") or DEFAULT_API_URL).strip()
