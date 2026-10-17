"""API для жюри (раздел 10 ТЗ) + раздача веб-интерфейса.

    uvicorn api.main:app --host 0.0.0.0 --port 8000
    curl -X POST http://localhost:8000/api/v1/analyze -H "X-API-Key: $API_KEY" -F "file=@video.mov"
    curl http://localhost:8000/api/v1/jobs/<job_id> -H "X-API-Key: $API_KEY"
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.pipeline import analyze  # noqa: E402

app = FastAPI(title="Cashier Compliance AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS_DIR = config.WORKDIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
# 1 воркер: модели на GPU лучше гонять последовательно; задачи встают в очередь
EXECUTOR = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()
_progress: dict[str, float] = {}


def require_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != config.API_KEY:
        raise HTTPException(401, "Неверный или отсутствующий X-API-Key")


def _save(job_id: str, data: dict):
    p = JOBS_DIR / job_id / "result.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # атомарно: GET никогда не прочитает полузаписанный файл


def _run(job_id: str, video: Path):
    try:
        res = analyze(video, progress=lambda p: _progress.__setitem__(job_id, p))
        _save(job_id, {"job_id": job_id, "status": "done", "verdict": res["verdict"],
                       "violations": res["violations"], "meta": res["_meta"]})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _save(job_id, {"job_id": job_id, "status": "failed", "verdict": None, "violations": [], "error": str(e)})
    finally:
        _progress.pop(job_id, None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def submit(file: UploadFile = File(...)):
    name = file.filename or "video.mov"
    if Path(name).suffix.lower() not in {".mov", ".mp4", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(415, "Ожидается видеофайл (.mov)")
    job_id = uuid.uuid4().hex[:8]
    d = JOBS_DIR / job_id
    d.mkdir(parents=True)
    video = d / f"input{Path(name).suffix.lower()}"
    with video.open("wb") as f:
        shutil.copyfileobj(file.file, f, length=1024 * 1024)
    if video.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "Файл слишком большой")
    _save(job_id, {"job_id": job_id, "status": "processing", "verdict": None, "violations": []})
    EXECUTOR.submit(_run, job_id, video)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def job(job_id: str):
    p = JOBS_DIR / job_id / "result.json"
    if not job_id.isalnum() or not p.exists():
        raise HTTPException(404, "Задача не найдена")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data["status"] == "processing":
        data["progress"] = round(_progress.get(job_id, 0.0), 2)
    return data


# ---- Веб-интерфейс (один HTML без сборки фронтенда) ----
UI = Path(__file__).resolve().parents[1] / "app" / "index.html"


@app.get("/")
def index():
    return FileResponse(UI)
