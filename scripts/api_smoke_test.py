"""Проверка API так, как его будет дёргать жюри (запускайте С ДРУГОЙ МАШИНЫ по публичному URL).

    python scripts/api_smoke_test.py https://your-url.example.kz open/videos/X-1.mov --key <API_KEY>
"""
import argparse
import json
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("base_url")
ap.add_argument("video")
ap.add_argument("--key", required=True)
a = ap.parse_args()
base, h = a.base_url.rstrip("/"), {"X-API-Key": a.key}

assert requests.post(f"{base}/api/v1/analyze", files={"file": open(a.video, "rb")}).status_code == 401, \
    "Запрос без ключа должен отклоняться!"
t0 = time.time()
r = requests.post(f"{base}/api/v1/analyze", headers=h, files={"file": open(a.video, "rb")}, timeout=120)
r.raise_for_status()
job = r.json()["job_id"]
print("job_id:", job, f"(загрузка {time.time() - t0:.1f} c)")
while True:
    j = requests.get(f"{base}/api/v1/jobs/{job}", headers=h, timeout=30).json()
    if j["status"] != "processing":
        break
    if time.time() - t0 > 300:
        raise SystemExit("❌ Не уложились в 5 минут — у жюри видео будет считаться необработанным")
    time.sleep(3)
print(f"status={j['status']} verdict={j.get('verdict')} за {time.time() - t0:.1f} c")
for v in j.get("violations", []):
    v = {**v, "evidence_frames": [f"<jpeg {len(f['image_base64']) // 1024} KB @ {f['timestamp_sec']} c>" for f in v["evidence_frames"]]}
    print(json.dumps(v, ensure_ascii=False, indent=2))
assert j["status"] == "done" and j["verdict"] in {"violation", "no_violation"}
print("✅ API соответствует схеме")
