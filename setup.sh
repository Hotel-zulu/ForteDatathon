#!/usr/bin/env bash
# macOS / Linux:  bash setup.sh
set -e
PY=${PY:-python3.11}
command -v $PY >/dev/null || PY=python3
$PY -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt -r requirements-extra.txt
python -m ipykernel install --user --name forte --display-name "Python (forte)" 2>/dev/null || true
cp -n .env.example .env 2>/dev/null || true
python scripts/check_env.py
echo "Готово. Активация: source .venv/bin/activate"
