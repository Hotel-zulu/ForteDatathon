"""Разметка Label Studio -> датасет для обучения детектора (txt: класс cx cy w h — для scripts/train_rtdetr.py).

    python scripts/label_studio.py export objects
    python scripts/label_studio.py export zones
    python scripts/build_dataset.py                 # -> datasets/cashier/
    python scripts/build_dataset.py --val X-3 X-18 X-30 X-44   # свои ролики для проверки

Берутся ТОЛЬКО кадры, которые разметили люди (Submit), плюс 42 кадра зон.
Разбиение на train/val — по роликам (а не по кадрам), иначе качество завышено.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotation"
OUT = ROOT / "datasets" / "cashier"
CLASSES = ["cash", "id_card", "phone", "banknote_counter", "laptop", "safe_closed", "safe_open",
           "door_closed", "door_open", "collector_bag", "personal_item", "food_drink",
           # новые классы (17.10): нужны для X-22, X-24, X-11/X-28, X-13/X-44/X-46
           "key_in_lock", "documents", "screen_on", "screen_locked", "cash_in_counter"]


def tasks(path):
    if not path.exists():
        return []
    out = []
    for t in json.loads(path.read_text(encoding="utf-8")):
        anns = [a for a in (t.get("annotations") or []) if not a.get("was_cancelled")]
        if anns:
            out.append((t, anns[-1]["result"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="*", help="ролики для валидации (по умолчанию ~20% случайно)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = tasks(ANN / "objects" / "export.json") + tasks(ANN / "zones" / "export.json")
    if not items:
        raise SystemExit("Нет размеченных кадров. Сначала: python scripts/label_studio.py export objects")
    videos = sorted({t["data"]["video"] for t, _ in items}, key=lambda v: int(v.split("-")[1]))
    if a.val:
        val = set(a.val)
    else:
        random.seed(a.seed)
        val = set(random.sample(videos, max(1, round(0.2 * len(videos)))))

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    cnt, frames = {"train": Counter(), "val": Counter()}, Counter()
    for t, result in items:
        split = "val" if t["data"]["video"] in val else "train"
        src = ROOT / t["data"]["image"].split("d=")[-1]
        name = f"{src.parent.parent.name}_{src.name}"  # zones_X-1.jpg / objects_X-1_010.000.jpg
        lines = []
        for r in result:
            if r.get("type") != "rectanglelabels":
                continue  # полигоны зон детектору не нужны
            v = r["value"]; lab = v["rectanglelabels"][0]
            if lab not in CLASSES:
                continue
            cx, cy = (v["x"] + v["width"] / 2) / 100, (v["y"] + v["height"] / 2) / 100
            w, h = v["width"] / 100, v["height"] / 100
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{CLASSES.index(lab)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            cnt[split][lab] += 1
        shutil.copy(src, OUT / "images" / split / name)
        (OUT / "labels" / split / name).with_suffix(".txt").write_text("\n".join(lines))
        frames[split] += 1

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
    print(f"Ролики для проверки (val): {sorted(val, key=lambda v: int(v.split('-')[1]))}")
    print(f"Кадров: train {frames['train']}, val {frames['val']}")
    print(f"{'класс':18s} {'train':>6s} {'val':>5s}")
    for c in CLASSES:
        warn = "  ⚠ мало примеров" if cnt["train"][c] < 30 else ""
        print(f"{c:18s} {cnt['train'][c]:6d} {cnt['val'][c]:5d}{warn}")
    print(f"\nДатасет: {OUT}/data.yaml")


if __name__ == "__main__":
    main()
