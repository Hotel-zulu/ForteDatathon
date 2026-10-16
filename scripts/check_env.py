"""Проверка окружения — пусть КАЖДЫЙ участник команды запустит это до хакатона.

    python scripts/check_env.py
"""
import importlib
import platform
import shutil
import subprocess
import sys

OK, FAIL, WARN = "✅", "❌", "⚠️ "
REQUIRED = ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "streamlit", "fastapi",
            "uvicorn", "joblib", "matplotlib", "pyarrow"]
OPTIONAL = ["optuna", "shap", "ydata_profiling", "jupyterlab", "torch", "transformers", "polars"]


def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return None if isinstance(e, ImportError) else f"ошибка: {e}"


def cmd(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return None


print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.machine()}")
if sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}:
    print(f"{WARN} Рекомендуется Python 3.10–3.12 (у CatBoost/LightGBM бывают проблемы с колёсами на новых версиях)")

bad = 0
print("\nОбязательные пакеты:")
for m in REQUIRED:
    v = ver(m)
    bad += v is None
    print(f"  {OK if v else FAIL} {m:12s} {v or 'НЕ УСТАНОВЛЕН'}")
print("\nОпциональные:")
for m in OPTIONAL:
    v = ver(m)
    print(f"  {OK if v else '·'} {m:16s} {v or '-'}")

print("\nИнструменты:")
for tool, args in [("git", ("git", "--version")), ("docker", ("docker", "--version")),
                   ("docker compose", ("docker", "compose", "version"))]:
    out = cmd(*args) if shutil.which(args[0]) else None
    print(f"  {OK if out else FAIL} {tool:15s} {out or 'не найден'}")
if shutil.which("docker"):
    running = cmd("docker", "info", "--format", "{{.ServerVersion}}")
    print(f"  {OK if running else WARN} docker daemon   {'запущен' if running else 'НЕ запущен (запустите Docker Desktop)'}")

print("\nGPU:")
gpu = cmd("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader") if shutil.which("nvidia-smi") else None
print(f"  {OK if gpu else '·'} {gpu or 'нет NVIDIA GPU — для NLP используйте Kaggle/Colab'}")
try:
    import torch
    print(f"  torch.cuda: {torch.cuda.is_available()}")
except ImportError:
    pass

print("\nИтог:", f"{OK} всё готово" if not bad else f"{FAIL} не хватает {bad} пакетов: pip install -r requirements.txt")
sys.exit(1 if bad else 0)
