#!/usr/bin/env bash
# macOS / Linux:  bash setup.sh        (нужен Python 3.10/3.11 и ffmpeg: brew install ffmpeg)
set -e
PY=${PY:-python3.11}; command -v $PY >/dev/null || PY=python3
$PY -m venv .venv && source .venv/bin/activate
pip install -U pip wheel && pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python scripts/check_env.py || true
python -m src.labels
echo "Готово. source .venv/bin/activate && make serve"
