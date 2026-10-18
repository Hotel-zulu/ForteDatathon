"""API для жюри (раздел 10 ТЗ) + данные для веб-панели.

    uvicorn api.main:app --host 0.0.0.0 --port 8000
    curl -X POST http://localhost:8000/api/v1/analyze -H "X-API-Key: $API_KEY" -F "file=@video.mov"
    curl http://localhost:8000/api/v1/jobs/<job_id> -H "X-API-Key: $API_KEY"

Эндпоинты жюри: POST /api/v1/analyze, GET /api/v1/jobs/{job_id} — формат по ТЗ.
Остальные (список, статистика, проверка человеком, нормы, качество) нужны только веб-панели.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from fastapi import Body, Cookie, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.pipeline import analyze  # noqa: E402

app = FastAPI(title="Cash Compliance AI", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS_DIR = config.WORKDIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
# 1 воркер: модели на GPU лучше гонять последовательно; задачи встают в очередь
EXECUTOR = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()
_progress: dict[str, float] = {}


# Веб-панель открывается без ввода ключа: при загрузке страницы сервер ставит HttpOnly-cookie,
# производную от API_KEY. Внешние запросы (curl, скрипты жюри) cookie не имеют — им нужен X-API-Key.
PANEL_COOKIE = "cc_panel"
PANEL_TOKEN = hmac.new(config.API_KEY.encode(), b"web-panel", hashlib.sha256).hexdigest()


def require_key(x_api_key: str | None = Header(default=None), key: str | None = Query(default=None),
                cc_panel: str | None = Cookie(default=None)):
    if config.API_KEY in (x_api_key, key):
        return
    if cc_panel and hmac.compare_digest(cc_panel, PANEL_TOKEN):
        return
    raise HTTPException(401, "Неверный или отсутствующий X-API-Key")


def _read(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(p: Path, data: dict):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # атомарно: GET никогда не прочитает полузаписанный файл


def _save(job_id: str, data: dict):
    _write(JOBS_DIR / job_id / "result.json", data)


def _job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    if not job_id.isalnum() or not (d / "result.json").exists():
        raise HTTPException(404, "Задача не найдена")
    return d


def _run(job_id: str, video: Path):
    try:
        res = analyze(video, progress=lambda p: _progress.__setitem__(job_id, p))
        _write(JOBS_DIR / job_id / "vision.json", {"frames": res.pop("_vision", [])})
        _save(job_id, {"job_id": job_id, "status": "done", "verdict": res["verdict"],
                       "violations": res["violations"], "meta": res["_meta"]})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _save(job_id, {"job_id": job_id, "status": "failed", "verdict": None, "violations": [], "error": str(e)})
    finally:
        _progress.pop(job_id, None)


@app.get("/health")
def health():
    import os
    w = Path(os.getenv("RTDETR_WEIGHTS", str(config.MODELS_DIR / "rtdetr_cashier")))
    return {"status": "ok", "engine": os.getenv("PERCEPTION", "rtdetr"), "local": True,
            "detector": "RT-DETRv2 (Apache-2.0)" if (w / "config.json").exists() else None, "queue": len(_progress)}


# ---------------- эндпоинты жюри ----------------
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
    _write(d / "info.json", {"file_name": Path(name).name, "created_at": time.time(),
                             "size_mb": round(video.stat().st_size / 1e6, 1)})
    _save(job_id, {"job_id": job_id, "status": "processing", "verdict": None, "violations": []})
    _progress[job_id] = 0.0
    EXECUTOR.submit(_run, job_id, video)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def job(job_id: str):
    data = _read(_job_dir(job_id) / "result.json")
    if data["status"] == "processing":
        data["progress"] = round(_progress.get(job_id, 0.0), 2)
    return data


# ---------------- данные для веб-панели ----------------
def _summary(d: Path) -> dict | None:
    r = _read(d / "result.json")
    if not r:
        return None
    info = _read(d / "info.json", {}) or {}
    meta = r.get("meta") or {}
    return {
        "job_id": d.name, "status": r["status"], "verdict": r.get("verdict"),
        "file_name": info.get("file_name") or "—",
        "created_at": info.get("created_at") or d.stat().st_mtime,
        "size_mb": info.get("size_mb"),
        "duration_sec": meta.get("duration_sec"), "processing_sec": meta.get("processing_sec"),
        "progress": round(_progress.get(d.name, 0.0), 2) if r["status"] == "processing" else None,
        "error": r.get("error"),
        "violations": [{"type": v["type"], "timecode_sec": v["timecode_sec"], "confidence": v.get("confidence")}
                       for v in r.get("violations", [])],
        "review": _read(d / "review.json"),
    }


def _all() -> list[dict]:
    out = [s for d in JOBS_DIR.iterdir() if d.is_dir() and (s := _summary(d))]
    return sorted(out, key=lambda s: -s["created_at"])


@app.get("/api/v1/jobs", dependencies=[Depends(require_key)])
def jobs_list(limit: int = 500):
    return _all()[:limit]


@app.get("/api/v1/stats", dependencies=[Depends(require_key)])
def stats(days: int = 7):
    js = _all()
    done = [j for j in js if j["status"] == "done"]
    viol = [j for j in done if j["verdict"] == "violation"]
    by_type = Counter(v["type"] for j in done for v in j["violations"])
    today = datetime.now().date()
    per_day = []
    for k in range(days - 1, -1, -1):
        day = today - timedelta(days=k)
        n = sum(1 for j in done if datetime.fromtimestamp(j["created_at"]).date() == day)
        per_day.append({"date": day.isoformat(), "count": n})
    reviewed = [j for j in viol if j["review"]]
    proc = [j["processing_sec"] for j in done if j["processing_sec"]]
    return {
        "total": len(js), "done": len(done), "processing": sum(j["status"] == "processing" for j in js),
        "failed": sum(j["status"] == "failed" for j in js),
        "with_violations": len(viol), "events": sum(len(j["violations"]) for j in done),
        "awaiting_review": len(viol) - len(reviewed),
        "confirmed": sum(1 for j in reviewed if j["review"].get("decision") == "confirmed"),
        "rejected": sum(1 for j in reviewed if j["review"].get("decision") == "rejected"),
        "avg_processing_sec": round(sum(proc) / len(proc), 1) if proc else None,
        "by_type": by_type.most_common(), "per_day": per_day,
    }


@app.post("/api/v1/jobs/{job_id}/review", dependencies=[Depends(require_key)])
def review(job_id: str, body: dict = Body(...)):
    """Решение сотрудника контроля: confirmed — нарушение подтверждено, rejected — ложное срабатывание."""
    d = _job_dir(job_id)
    if body.get("decision") not in ("confirmed", "rejected", None):
        raise HTTPException(422, "decision: confirmed | rejected | null")
    if body.get("decision") is None:
        (d / "review.json").unlink(missing_ok=True)
        return {"ok": True}
    rec = {"decision": body["decision"], "note": str(body.get("note", ""))[:1000],
           "reviewer": str(body.get("reviewer", ""))[:100], "at": time.time()}
    _write(d / "review.json", rec)
    return rec


@app.get("/api/v1/jobs/{job_id}/video", dependencies=[Depends(require_key)])
def job_video(job_id: str):
    vids = list(_job_dir(job_id).glob("input.*"))
    if not vids:
        raise HTTPException(404, "Видео не сохранено")
    return FileResponse(vids[0])


@app.get("/api/v1/jobs/{job_id}/vision", dependencies=[Depends(require_key)])
def job_vision(job_id: str):
    """Покадровые детекции модели (режим «Как видит ИИ» в панели)."""
    return _read(_job_dir(job_id) / "vision.json", {"frames": []})


@app.get("/api/v1/regulations", dependencies=[Depends(require_key)])
def regulations():
    cfg = yaml.safe_load(config.REGULATIONS.read_text(encoding="utf-8"))
    types = []
    for tid, v in cfg["violation_types"].items():
        src = cfg["sources"][v["source"]]
        types.append({"id": tid, "name": v["name"], "enabled": bool(v.get("enabled")), "clause": v["clause"],
                      "quote": " ".join(str(v["quote"]).split()), "observable": v.get("observable"),
                      "note": v.get("note"), "extra_ref": v.get("extra_ref"), "videos": v.get("videos", []),
                      "verified": bool(v.get("verified")), "act": src["act"], "url": src["url"],
                      "title": " ".join(src["title"].split())})
    return {"types": types}


@app.get("/api/v1/benchmark", dependencies=[Depends(require_key)])
def benchmark():
    """Качество на открытых роликах (scripts/eval_timelines.py пишет workdir/eval/latest.json)."""
    return _read(config.WORKDIR / "eval" / "latest.json", {"available": False})


# ---- Веб-панель (один HTML без сборки и без интернета) ----
UI = ROOT / "app" / "index.html"


@app.get("/")
def index():
    r = FileResponse(UI, headers={"Cache-Control": "no-store"})
    r.set_cookie(PANEL_COOKIE, PANEL_TOKEN, httponly=True, samesite="strict", max_age=7 * 24 * 3600)
    return r
