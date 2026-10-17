"""Проверка окружения (запустить каждому участнику):  python scripts/check_env.py"""
import importlib
import platform
import shutil
import subprocess
import sys

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "cv2": "opencv-python-headless", "yaml": "pyyaml",
            "openpyxl": "openpyxl", "pydantic": "pydantic", "fastapi": "fastapi", "uvicorn": "uvicorn",
            "multipart": "python-multipart", "requests": "requests"}
OPTIONAL = ["torch", "torchvision", "transformers", "supervision", "onnxruntime"]


def ver(m):
    try:
        return getattr(importlib.import_module(m), "__version__", "ok")
    except Exception:  # noqa: BLE001
        return None


print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.machine()}")
bad = []
print("\nОбязательные:")
for mod, pkg in REQUIRED.items():
    v = ver(mod)
    print(f"  {'✅' if v else '❌'} {pkg:24s} {v or 'НЕ УСТАНОВЛЕН'}")
    if not v:
        bad.append(pkg)
print("\nМодели (опционально):")
for m in OPTIONAL:
    print(f"  {'✅' if ver(m) else '·'} {m:24s} {ver(m) or '-'}")

print("\nИнструменты:")
for tool in ["ffmpeg", "git", "docker", "cloudflared"]:
    p = shutil.which(tool)
    print(f"  {'✅' if p else '·'} {tool:12s} {p or 'не найден'}")

print("\nУскорители:")
try:
    import torch
    print(f"  CUDA: {torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"  Apple MPS: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
except ImportError:
    gpu = shutil.which("nvidia-smi") and subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    print(f"  {gpu or 'torch не установлен'}")

if ver("cv2"):
    import cv2
    print(f"\nOpenCV backends: FFMPEG={'FFMPEG' in cv2.getBuildInformation()}")

print("\nИтог:", "✅ готово" if not bad else f"❌ pip install {' '.join(bad)}")
sys.exit(1 if bad else 0)
