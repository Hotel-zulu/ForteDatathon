"""Управление локальным Label Studio для разметки.

    python scripts/label_studio.py start          # запустить сервер (http://localhost:8080)
    python scripts/label_studio.py setup zones    # создать проект «1. Зоны» и загрузить задачи
    python scripts/label_studio.py setup objects  # создать проект «2. Объекты» и загрузить задачи
    python scripts/label_studio.py export zones   # выгрузить разметку -> annotation/zones/export.json
    python scripts/label_studio.py export objects # -> annotation/objects/export.json

Логин в браузере: см. annotation/.ls_credentials (создаётся при первом запуске).
Всё работает локально: кадры не покидают компьютер.
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotation"
CRED = ANN / ".ls_credentials"
URL = os.getenv("LS_URL", "http://localhost:8080")
LS_BIN = Path.home() / "label-studio-venv" / "bin" / "label-studio"
TITLES = {"zones": "1. Зоны (по кадру на видео)", "objects": "2. Объекты (детектор)"}


def creds() -> dict:
    if CRED.exists():
        return json.loads(CRED.read_text())
    c = {"email": "team@forte.local", "password": secrets.token_urlsafe(10), "token": secrets.token_hex(20)}
    ANN.mkdir(exist_ok=True)
    CRED.write_text(json.dumps(c, indent=1))
    return c


def start():
    c = creds()
    env = {**os.environ,
           "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
           "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(ROOT),
           "LABEL_STUDIO_USERNAME": c["email"], "LABEL_STUDIO_PASSWORD": c["password"],
           "LABEL_STUDIO_USER_TOKEN": c["token"],
           "LABEL_STUDIO_BASE_DATA_DIR": str(ANN / ".ls_data"),
           "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK": "true"}
    public = os.getenv("LS_PUBLIC_URL")  # адрес туннеля (https://....trycloudflare.com), если есть
    if public:
        env.update({"LABEL_STUDIO_HOST": public, "CSRF_TRUSTED_ORIGINS": public,
                    "LABEL_STUDIO_CSRF_TRUSTED_ORIGINS": public})
        print(f"Публичный адрес: {public}")
    print(f"Label Studio: {URL}\nЛогин: {c['email']}\nПароль: {c['password']}")
    subprocess.run([str(LS_BIN), "start", "--port", URL.rsplit(":", 1)[-1], "--no-browser"], env=env, check=False)


_SESSION: requests.Session | None = None


def session() -> requests.Session:
    """Вход логином/паролем (в LS 1.2x старые API-токены отключены по умолчанию)."""
    global _SESSION
    if _SESSION is None:
        s, c = requests.Session(), creds()
        s.get(URL + "/user/login/", timeout=30)
        r = s.post(URL + "/user/login/", data={"email": c["email"], "password": c["password"],
                                               "csrfmiddlewaretoken": s.cookies.get("csrftoken", "")},
                   headers={"Referer": URL + "/user/login/"}, timeout=30, allow_redirects=True)
        if "sessionid" not in s.cookies:
            raise SystemExit(f"Не удалось войти в Label Studio ({r.status_code}). Сервер запущен?")
        _SESSION = s
    return _SESSION


def api(method, path, **kw):
    s = session()
    h = {"X-CSRFToken": s.cookies.get("csrftoken", ""), "Referer": URL + "/"}
    r = s.request(method, URL + path, headers=h, timeout=120, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.content else {}


def find_project(title):
    res = api("GET", "/api/projects", params={"page_size": 100})
    items = res.get("results", res) if isinstance(res, dict) else res
    return next((p for p in items if p["title"] == title), None)


def setup(stage):
    tasks_path = ANN / stage / "tasks.json"
    if not tasks_path.exists():
        raise SystemExit(f"Нет {tasks_path}. Сначала: python scripts/annotation_prepare.py {stage}")
    title = TITLES[stage]
    if find_project(title):
        raise SystemExit(f"Проект «{title}» уже существует — удалите его в интерфейсе, если хотите пересоздать")
    cfg = (ANN / f"config_{stage}.xml").read_text(encoding="utf-8")
    p = api("POST", "/api/projects", json={"title": title, "label_config": cfg,
                                            "show_collab_predictions": True})
    api("POST", "/api/storages/localfiles", json={
        "project": p["id"], "title": f"{stage} images", "path": str(ANN / stage / "images"),
        "use_blob_urls": False, "regex_filter": r".*\.jpg"})
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    for i in range(0, len(tasks), 200):
        api("POST", f"/api/projects/{p['id']}/import", json=tasks[i:i + 200])
    print(f"✅ Проект «{title}»: {len(tasks)} задач -> {URL}/projects/{p['id']}")


def export(stage):
    p = find_project(TITLES[stage])
    if not p:
        raise SystemExit("Проект не найден")
    data = api("GET", f"/api/projects/{p['id']}/export", params={"exportType": "JSON"})
    out = ANN / stage / "export.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    done = sum(1 for t in data if t.get("annotations"))
    print(f"Выгружено {len(data)} задач (размечено {done}) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "start":
        start()
    elif cmd in {"setup", "export"} and len(sys.argv) > 2:
        {"setup": setup, "export": export}[cmd](sys.argv[2])
    else:
        print(__doc__)
