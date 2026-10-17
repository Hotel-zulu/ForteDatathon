"""Настройки сервиса. Всё переопределяется переменными окружения / файлом .env."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

API_KEY = os.getenv("API_KEY", "change-me")              # ключ для жюри (заголовок X-API-Key)
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "10"))        # прореживание кадров для анализа
MAX_SIDE = int(os.getenv("MAX_SIDE", "960"))             # ресайз кадра для моделей (длинная сторона)
TIME_BUDGET_SEC = float(os.getenv("TIME_BUDGET_SEC", "240"))  # лимит жюри 300 c — оставляем запас
BLUR_FACES = os.getenv("BLUR_FACES", "1") == "1"         # размывать лица на кадрах-доказательствах
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "300"))

VIDEOS_DIR = ROOT / "open" / "videos"
LABELS_XLSX = ROOT / "open" / "open_labels.xlsx"
LABELS_CSV = ROOT / "configs" / "labels.csv"
REGULATIONS = ROOT / "configs" / "regulations.yaml"
WORKDIR = ROOT / "workdir"            # загруженные видео и результаты задач
MODELS_DIR = ROOT / "models"          # веса моделей (кладутся заранее — сервис работает офлайн)
