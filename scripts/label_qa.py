"""Проверка разметки: где модель уверенно не согласна с людьми -> список кадров для перепроверки.

    python scripts/label_studio.py export objects
    python scripts/label_qa.py              # отчёт + пометка задач в Label Studio (поле check = «QA: …»)
    python scripts/label_qa.py --dry-run    # только отчёт workdir/label_qa.csv

В Label Studio: Data Manager → фильтр по колонке check содержит «QA» → открыть, поправить, Submit.
Типы подсказок:
  «нет рамки X?»     — модель уверенно видит X (≥0.6), а рамки X рядом нет (пропуск при разметке);
  «X или Y?»         — на месте рамки X модель уверенно видит другой класс Y (перепутана метка);
  «лишняя рамка X?»  — под рамкой X модель не видит ничего похожего (ошибочная/сдвинутая рамка).
Это подсказки, а не истина: решает человек.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
STATIC = {"banknote_counter", "laptop", "safe_closed", "door_closed"}   # почти всегда верны — не шумим
RARE = {"documents", "personal_item", "cash_in_counter", "collector_bag", "safe_open", "id_card", "key_in_lock",
        "door_open", "screen_on", "screen_locked", "cash"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / (u + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "rtdetr_cashier"))
    ap.add_argument("--top", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import torch
    from src.detectors.rtdetr_perception import _Det
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = _Det(a.weights, dev, 640)

    tasks = json.loads((ROOT / "annotation" / "objects" / "export.json").read_text(encoding="utf-8"))
    rows = []
    for t in tasks:
        anns = [x for x in (t.get("annotations") or []) if not x.get("was_cancelled")]
        if not anns:
            continue
        gt = [(r["value"]["rectanglelabels"][0],
               [r["value"]["x"] / 100, r["value"]["y"] / 100, (r["value"]["x"] + r["value"]["width"]) / 100,
                (r["value"]["y"] + r["value"]["height"]) / 100])
              for r in anns[-1]["result"] if r.get("type") == "rectanglelabels"]
        img = np.asarray(Image.open(ROOT / t["data"]["image"].split("d=")[-1]).convert("RGB"))
        pred = model(img, 0.1)
        issues, score = [], 0.0
        for n, s, b in pred:                       # пропущенная рамка
            if s >= 0.6 and n not in STATIC and not any(g[0] == n and iou(g[1], b) > 0.3 for g in gt):
                issues.append(f"нет рамки {n}?"); score += s * (1.5 if n in RARE else 1)
        for gn, gb in gt:
            if gn in STATIC:
                continue
            same = max([s for n, s, b in pred if n == gn and iou(b, gb) > 0.3] or [0])
            other = [(s, n) for n, s, b in pred if n != gn and s >= 0.6 and iou(b, gb) > 0.5]
            if other and same < 0.3:              # перепутан класс
                s, n = max(other)
                issues.append(f"{gn} или {n}?"); score += s * 1.5
            elif same < 0.1:                       # рамка без опоры
                issues.append(f"лишняя рамка {gn}?"); score += 0.5
        if issues:
            rows.append({"task": t["id"], "video": t["data"].get("video"), "image": Path(t["data"]["image"]).name,
                         "score": round(score, 2), "issues": "; ".join(sorted(set(issues)))})
    rows.sort(key=lambda r: -r["score"])
    top = rows[:a.top]
    out = ROOT / "workdir" / "label_qa.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["task", "video", "image", "score", "issues"])
        w.writeheader(); w.writerows(top)
    kinds = Counter(i.split(" ")[0] + (" " + i.split(" ")[1] if i.startswith("нет") or i.startswith("лишняя") else "")
                    for r in top for i in r["issues"].split("; "))
    print(f"Подозрительных кадров: {len(rows)}, в списке топ-{len(top)} -> {out}")
    print("По видео:", Counter(r["video"] for r in top).most_common(12))
    if not a.dry_run:
        import label_studio as L
        topset = {r["task"]: r["issues"] for r in top}
        for t in tasks:
            if t["id"] in topset:
                data = dict(t["data"]); data["check"] = "QA: " + topset[t["id"]]
                L.api("PATCH", f"/api/tasks/{t['id']}", json={"data": data})
        print("В Label Studio помечено поле check = «QA: …»")


if __name__ == "__main__":
    main()
