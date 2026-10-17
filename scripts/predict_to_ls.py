"""Черновики от ОБУЧЕННОЙ модели во все неразмеченные кадры «2. Объекты» (RT-DETR, Apache-2.0).

    python scripts/predict_to_ls.py --dry-run
    python scripts/predict_to_ls.py                       # модель: models/rtdetr_cashier

Для каждого неразмеченного кадра:
  • неподвижные объекты (счётчик, ноутбук, сейф, дверь) — рамки с кадра зоны (точные);
    состояние сейфа/двери берётся у модели, если она уверена (open/closed), иначе остаётся из зоны
    + пометка check, если участок заметно изменился;
  • подвижные объекты (деньги, удостоверение, телефон, сумки, еда) — из модели, без зоны стекла.
Разметку людей не трогает. Версия черновиков: model-v1 (номер можно задать --version).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import label_studio as L  # noqa: E402
from owl_prelabel import inside, iou, load_glass  # noqa: E402
from update_drafts import all_tasks, changed, crop_sig, load_zones  # noqa: E402

MOVERS = {"cash", "id_card", "phone", "collector_bag", "personal_item", "food_drink"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "rtdetr_cashier"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--state-conf", type=float, default=0.45, help="уверенность для open/closed сейфа и двери")
    ap.add_argument("--version", default="model-v1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import torch
    from src.detectors.rtdetr_perception import _Det
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = _Det(a.weights, dev, 640)
    zones, glass = load_zones(), load_glass()
    zone_sig = {v: [crop_sig(Image.open(z["image"]).convert("RGB"), r) for r in z["rects"]] for v, z in zones.items()}

    proj = L.find_project(L.TITLES["objects"])
    stats, n = Counter(), 0
    for t in all_tasks(proj["id"]):
        if t.get("total_annotations") or t.get("annotations"):
            continue
        vid = t["data"].get("video")
        img = Image.open(ROOT / t["data"]["image"].split("d=")[-1]).convert("RGB")
        W, H = img.size
        dets = [{"cls": n, "score": sc, "box": (b[0] * W, b[1] * H, b[2] * W, b[3] * H)}
                for n, sc, b in model(np.asarray(img), a.conf)]
        # без отражений
        dets = [d for d in dets if not any(inside((100 * (d["box"][0] + d["box"][2]) / 2 / W,
                                                   100 * (d["box"][1] + d["box"][3]) / 2 / H), p)
                                           for p in glass.get(vid, []))]
        result, check = [], []
        for i, v in enumerate(zones.get(vid, {}).get("rects", [])):
            v = dict(v); lab = v["rectanglelabels"][0]
            zb = (v["x"] / 100 * W, v["y"] / 100 * H, (v["x"] + v["width"]) / 100 * W, (v["y"] + v["height"]) / 100 * H)
            kind = "safe" if lab.startswith("safe") else "door" if lab.startswith("door") else None
            if kind:
                cand = [d for d in dets if d["cls"].startswith(kind) and d["score"] >= a.state_conf and iou(zb, d["box"]) > 0.3]
                if cand:
                    v["rectanglelabels"] = [max(cand, key=lambda d: d["score"])["cls"]]
                ch, _ = changed(crop_sig(img, v), zone_sig[vid][i], 0.55)
                if ch or v["rectanglelabels"][0].endswith("_open"):
                    check.append("сейф" if kind == "safe" else "дверь")
                stats[v["rectanglelabels"][0]] += 1
            result.append({"id": f"z{t['id']}_{i}", "from_name": "box", "to_name": "image", "type": "rectanglelabels",
                           "original_width": W, "original_height": H, "image_rotation": 0, "value": v})
        for j, d in enumerate(x for x in dets if x["cls"] in MOVERS):
            x1, y1, x2, y2 = d["box"]
            result.append({"id": f"m{t['id']}_{j}", "from_name": "box", "to_name": "image", "type": "rectanglelabels",
                           "original_width": W, "original_height": H, "image_rotation": 0, "score": float(d["score"]),
                           "value": {"x": 100 * x1 / W, "y": 100 * y1 / H, "width": 100 * (x2 - x1) / W,
                                     "height": 100 * (y2 - y1) / H, "rotation": 0, "rectanglelabels": [d["cls"]]}})
            stats[d["cls"]] += 1
        if not a.dry_run:
            for p in L.api("GET", "/api/predictions", params={"task": t["id"]}):
                L.api("DELETE", f"/api/predictions/{p['id']}")
            L.api("POST", "/api/predictions", json={"task": t["id"], "model_version": a.version, "result": result})
            data = dict(t["data"]); data["check"] = ", ".join(sorted(set(check)))
            L.api("PATCH", f"/api/tasks/{t['id']}", json={"data": data})
        n += 1
        if n % 100 == 0:
            print(f"  … {n} кадров", flush=True)
    if not a.dry_run:
        L.api("PATCH", f"/api/projects/{proj['id']}", json={"model_version": a.version})
    print(f"\nГотово{' (пробный прогон)' if a.dry_run else ''}: {n} кадров")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
