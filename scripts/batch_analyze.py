"""Отправить все открытые ролики в сервис через API (как жюри) — заполняет историю веб-панели реальными анализами.

    python scripts/batch_analyze.py                         # localhost:8000, ключ из .env
    python scripts/batch_analyze.py --only X-26 X-11 --url https://…trycloudflare.com
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.labels import norm_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default=config.API_KEY)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    h = {"X-API-Key": a.key}
    vids = sorted(config.VIDEOS_DIR.glob("*.mov"), key=lambda p: int(norm_name(p.stem).split("-")[1]))
    if a.only:
        keep = {norm_name(x) for x in a.only}
        vids = [v for v in vids if norm_name(v.stem) in keep]
    for v in vids:
        t0 = time.time()
        with v.open("rb") as f:   # имя файла латиницей: X-26.mov
            r = requests.post(f"{a.url}/api/v1/analyze", headers=h,
                              files={"file": (norm_name(v.stem) + ".mov", f, "video/quicktime")}, timeout=600)
        r.raise_for_status()
        jid = r.json()["job_id"]
        while True:
            j = requests.get(f"{a.url}/api/v1/jobs/{jid}", headers=h, timeout=60).json()
            if j["status"] != "processing":
                break
            time.sleep(2)
        types = ", ".join(f"{x['type']} @{x['timecode_sec']}" for x in j.get("violations", []))
        print(f"{norm_name(v.stem):6s} {j['status']:6s} {j.get('verdict') or '':13s} {time.time() - t0:5.1f} c  {types}", flush=True)


if __name__ == "__main__":
    main()
