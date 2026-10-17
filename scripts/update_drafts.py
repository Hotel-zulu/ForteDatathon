"""Обновить черновики во всех НЕРАЗМЕЧЕННЫХ кадрах проекта «2. Объекты» после разметки зон.

    python scripts/label_studio.py export zones
    python scripts/update_drafts.py --dry-run     # только посчитать, ничего не менять
    python scripts/update_drafts.py               # применить

Что делает для каждого кадра (разметку людей не трогает):
  1. Неподвижные объекты (счётчик, ноутбук, сейф, дверь) копирует с кадра зоны того же видео.
  2. Сравнивает участки сейфа и двери с кадром зоны. Если картинка заметно изменилась —
     помечает кадр `check` = «сейф» / «дверь» (фильтр в Data Manager), а для сейфа ставит
     safe_open, если OWLv2 видит там открытую дверцу.
  3. Добавляет черновики OWLv2 (деньги, удостоверение, телефон, сумки, еда) и YOLO (телефон, стакан),
     отбрасывая всё, что попало в полигон стекла (отражения).
  Старые черновики заменяются одним новым набором «draft-v2».
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import label_studio as L  # noqa: E402
from owl_prelabel import Owl, iou, load_glass, refine  # noqa: E402

ZONES_EXPORT = ROOT / "annotation" / "zones" / "export.json"
STATIC = {"banknote_counter", "laptop", "safe_open", "safe_closed", "door_open", "door_closed"}
VERSION = "draft-v2"


def load_zones():
    zones = {}
    for t in json.loads(ZONES_EXPORT.read_text(encoding="utf-8")):
        anns = t.get("annotations") or []
        if not anns:
            continue
        rects = [r["value"] for r in anns[-1]["result"]
                 if r.get("type") == "rectanglelabels" and r["value"]["rectanglelabels"][0] in STATIC]
        zones[t["data"]["video"]] = {"rects": rects, "image": ROOT / t["data"]["image"].split("d=")[-1]}
    return zones


def crop_sig(img: Image.Image, v: dict) -> np.ndarray:
    W, H = img.size
    box = (v["x"] / 100 * W, v["y"] / 100 * H, (v["x"] + v["width"]) / 100 * W, (v["y"] + v["height"]) / 100 * H)
    g = np.asarray(img.crop(box).convert("L").resize((32, 32)), dtype=np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def changed(a: np.ndarray, b: np.ndarray, thr: float) -> tuple[bool, float]:
    d = float(np.abs(a - b).mean())
    return d > thr, d


def yolo_dets(img):
    try:
        from ultralytics import YOLO
        from src.detectors.yolo_perception import pick_device
    except ImportError:
        return []
    global _Y
    if "_Y" not in globals():
        _Y = (YOLO(str(ROOT / "models" / "yolo11s.pt")), pick_device())
    m, dev = _Y
    r = m.predict(np.asarray(img)[:, :, ::-1], imgsz=1280, conf=0.35, classes=[67], device=dev, verbose=False)[0]
    names = {67: "phone"}  # стакан/бутылку не ищем: давали ложные срабатывания на деньгах
    return [{"cls": names[c], "score": s, "box": tuple(b)}
            for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist())]


def all_tasks(pid):
    page, out = 1, []
    while True:
        r = L.api("GET", "/api/tasks", params={"project": pid, "page": page, "page_size": 100, "fields": "all"})
        tasks = r.get("tasks", r) if isinstance(r, dict) else r
        if not tasks:
            break
        out += tasks
        if len(tasks) < 100:
            break
        page += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--thr-safe", type=float, default=0.55)
    ap.add_argument("--thr-door", type=float, default=0.55)
    ap.add_argument("--only", nargs="*", help="только эти видео, напр. X-1 X-2")
    a = ap.parse_args()

    zones, glass = load_zones(), load_glass()
    print(f"Зоны размечены для {len(zones)} видео")
    zone_sig = {}
    for vid, z in zones.items():
        zi = Image.open(z["image"]).convert("RGB")
        zone_sig[vid] = [crop_sig(zi, v) for v in z["rects"]]

    owl = Owl()
    proj = L.find_project(L.TITLES["objects"])
    tasks = all_tasks(proj["id"])
    stats, flagged, diffs = Counter(), Counter(), {"safe": [], "door": []}
    for t in tasks:
        vid = t["data"].get("video")
        if a.only and vid not in a.only:
            continue
        if t.get("total_annotations") or t.get("annotations"):
            stats["пропущено (уже размечено)"] += 1
            continue
        img = Image.open(ROOT / t["data"]["image"].split("d=")[-1]).convert("RGB")
        W, H = img.size
        dets = refine(owl(img), img, vid, glass)
        result, check = [], []
        owl_safe_open = [d for d in dets if d["cls"] == "safe_open"]

        # 1–2. неподвижные объекты + изменения состояния
        for i, v in enumerate(zones.get(vid, {}).get("rects", [])):
            v = dict(v); lab = v["rectanglelabels"][0]
            kind = "safe" if lab.startswith("safe") else "door" if lab.startswith("door") else None
            if kind:
                ch, d = changed(crop_sig(img, v), zone_sig[vid][i], a.thr_safe if kind == "safe" else a.thr_door)
                diffs[kind].append(d)
                if ch:
                    check.append("сейф" if kind == "safe" else "дверь")
                    if kind == "safe":
                        zb = (v["x"] / 100 * W, v["y"] / 100 * H, (v["x"] + v["width"]) / 100 * W, (v["y"] + v["height"]) / 100 * H)
                        if any(iou(zb, o["box"]) > 0.1 for o in owl_safe_open):
                            v["rectanglelabels"] = ["safe_open"]
            result.append({"id": f"s{t['id']}_{i}", "from_name": "box", "to_name": "image", "type": "rectanglelabels",
                           "original_width": W, "original_height": H, "image_rotation": 0, "value": v})
        if vid not in zones:
            stats["нет зоны для видео"] += 1

        # 3. подвижные объекты: OWLv2 + YOLO (без дублей)
        movers = [d for d in dets if d["cls"] != "safe_open"]
        for y in refine(yolo_dets(img), img, vid, glass):
            if all(iou(y["box"], m["box"]) < 0.4 for m in movers):
                movers.append(y)
        for j, d in enumerate(movers):
            x1, y1, x2, y2 = d["box"]
            result.append({"id": f"m{t['id']}_{j}", "from_name": "box", "to_name": "image", "type": "rectanglelabels",
                           "original_width": W, "original_height": H, "image_rotation": 0, "score": float(d["score"]),
                           "value": {"x": 100 * x1 / W, "y": 100 * y1 / H, "width": 100 * (x2 - x1) / W,
                                     "height": 100 * (y2 - y1) / H, "rotation": 0, "rectanglelabels": [d["cls"]]}})
            stats[f"черновик {d['cls']}"] += 1
        for c in check:
            flagged[c] += 1
        stats["обновлено кадров"] += 1

        if not a.dry_run:
            for p in L.api("GET", "/api/predictions", params={"task": t["id"]}):
                L.api("DELETE", f"/api/predictions/{p['id']}")
            L.api("POST", "/api/predictions", json={"task": t["id"], "model_version": VERSION, "result": result})
            data = dict(t["data"]); data["check"] = ", ".join(check) if check else ""
            L.api("PATCH", f"/api/tasks/{t['id']}", json={"data": data})
        if stats["обновлено кадров"] % 100 == 0:
            print(f"  … {stats['обновлено кадров']} кадров")

    if not a.dry_run:
        L.api("PATCH", f"/api/projects/{proj['id']}", json={"model_version": VERSION})
    print("\n=== Итог", "(ПРОБНЫЙ ПРОГОН, ничего не изменено)" if a.dry_run else "")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  помечено «проверить»: {dict(flagged)}")
    for k, arr in diffs.items():
        if arr:
            q = np.percentile(arr, [50, 90, 99]).round(2)
            print(f"  отличие участка «{k}» от зоны: медиана {q[0]}, 90% {q[1]}, 99% {q[2]}")


if __name__ == "__main__":
    main()
