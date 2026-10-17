"""Подготовка кадров и черновой разметки для Label Studio.

Этап 1 — ЗОНЫ (один кадр на видео, 42 кадра):
    python scripts/annotation_prepare.py zones
  -> annotation/zones/images/*.jpg + annotation/zones/tasks.json
  Люди размечают: пол комнаты, стеклянную стену, стол (полигоны) и неподвижные объекты
  (счётчик банкнот, ноутбук, сейф, дверь — прямоугольники с состоянием open/closed).

Этап 2 — ОБЪЕКТЫ (~600–900 кадров):
    python scripts/annotation_prepare.py objects --zones-export annotation/zones/export.json
  -> annotation/objects/images/*.jpg + annotation/objects/tasks.json
  Черновик уже содержит: рамки нашей модели RT-DETR и рамки неподвижных
  объектов, скопированные с этапа 1 (камера в ролике неподвижна). Людям остаётся
  добавить деньги/удостоверение/сумку инкассатора и поправить состояние сейфа/двери.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import VIDEOS_DIR  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.video_io import frame_at, iter_frames, probe  # noqa: E402

ANN = ROOT / "annotation"
STATIC = {"banknote_counter", "laptop", "safe_open", "safe_closed", "door_open", "door_closed"}
# COCO -> наши классы (черновик, люди поправят)
COCO_MAP = {67: "phone", 63: "laptop", 41: "food_drink", 39: "food_drink", 40: "food_drink",
            26: "personal_item", 24: "personal_item", 28: "personal_item"}


def ls_url(path: Path) -> str:
    """Ссылка на локальный файл для Label Studio (корень = папка проекта)."""
    return "/data/local-files/?d=" + str(path.relative_to(ROOT)).replace("\\", "/")


def rect(label, x1, y1, x2, y2, w, h, rid, score=None):
    r = {"id": rid, "from_name": "box", "to_name": "image", "type": "rectanglelabels",
         "original_width": w, "original_height": h, "image_rotation": 0,
         "value": {"x": 100 * x1 / w, "y": 100 * y1 / h, "width": 100 * (x2 - x1) / w,
                   "height": 100 * (y2 - y1) / h, "rotation": 0, "rectanglelabels": [label]}}
    if score is not None:
        r["score"] = float(score)
    return r


class Prelabeler:
    """Черновые рамки от нашей модели RT-DETR (Apache-2.0), если она уже обучена; иначе — без черновиков."""

    def __init__(self, conf=0.35):
        try:
            import torch
            from src.detectors.rtdetr_perception import _Det
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = _Det(str(ROOT / "models" / "rtdetr_cashier"), dev, 640)
        except Exception as e:  # noqa: BLE001
            print(f"[prelabel] модель недоступна ({e}) — черновик без авторазметки")
            self.model = None
        self.conf = conf

    def __call__(self, img, prefix):
        if self.model is None:
            return []
        h, w = img.shape[:2]
        return [rect(name, b[0] * w, b[1] * h, b[2] * w, b[3] * h, w, h, f"{prefix}_{i}", sc)
                for i, (name, sc, b) in enumerate(self.model(img[:, :, ::-1].copy(), self.conf))]


def videos():
    lab = load_labels()
    lab = lab[lab.file.notna()].copy()
    lab["n"] = lab.key.str.extract(r"(\d+)")[0].astype(int)
    return lab.sort_values("n")


def write_tasks(tasks, out):
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Задач: {len(tasks)} -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- этап 1
def stage_zones(a):
    d = ANN / "zones" / "images"
    d.mkdir(parents=True, exist_ok=True)
    pre = Prelabeler()
    tasks = []
    for r in videos().itertuples():
        m = probe(VIDEOS_DIR / r.file)
        t = min(1.0, m.duration_sec / 2)  # в начале ролика обычно ничего не загораживает
        img = frame_at(VIDEOS_DIR / r.file, t)
        p = d / f"{r.key}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        pred = [x for x in pre(img, r.key) if x["value"]["rectanglelabels"][0] in STATIC]
        tasks.append({"data": {"image": ls_url(p), "video": r.key, "t": t},
                      "predictions": [{"model_version": "rtdetr-draft", "result": pred}] if pred else []})
        print(f"{r.key}: {len(pred)} авторамок")
    write_tasks(tasks, ANN / "zones" / "tasks.json")


# ---------------------------------------------------------------- этап 2
def load_static_from_zones(path: Path):
    """Экспорт Label Studio (JSON) -> {video: [прямоугольники неподвижных объектов]}"""
    if not path or not path.exists():
        print("⚠ Экспорт зон не найден — неподвижные объекты не будут скопированы")
        return {}
    out = defaultdict(list)
    for t in json.loads(path.read_text(encoding="utf-8")):
        key = t["data"].get("video")
        anns = t.get("annotations") or []
        if not key or not anns:
            continue
        for res in anns[-1]["result"]:
            if res.get("type") == "rectanglelabels" and res["value"]["rectanglelabels"][0] in STATIC:
                out[key].append(res["value"])
    print(f"Неподвижные объекты взяты из зон для {len(out)} видео")
    return out


def stage_objects(a):
    d = ANN / "objects" / "images"
    d.mkdir(parents=True, exist_ok=True)
    static = load_static_from_zones(Path(a.zones_export) if a.zones_export else None)
    pre = Prelabeler()
    tasks, total = [], 0
    for r in videos().itertuples():
        tc = r.timecode_sec if r.violation and pd.notna(r.timecode_sec) else None
        last_kept, last_thumb, n = -99.0, None, 0
        for t, img in iter_frames(VIDEOS_DIR / r.file, target_fps=4):
            dense = tc is not None and abs(t - tc) <= a.window
            step = a.dense_step if dense else a.step
            if t - last_kept < step - 1e-6:
                continue
            thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (36, 64)).astype(np.float32)
            if not dense and last_thumb is not None and np.abs(thumb - last_thumb).mean() < a.dedup:
                continue  # почти тот же кадр — пропускаем
            last_kept, last_thumb = t, thumb
            h, w = img.shape[:2]
            name = f"{r.key}_{t:07.3f}.jpg"
            p = d / name
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            prefix = f"{r.key}_{int(t * 1000)}"
            result = [x for x in pre(img, prefix) if x["value"]["rectanglelabels"][0] not in STATIC]
            for i, v in enumerate(static.get(r.key, [])):
                result.append({"id": f"{prefix}_s{i}", "from_name": "box", "to_name": "image",
                               "type": "rectanglelabels", "original_width": w, "original_height": h,
                               "image_rotation": 0, "value": dict(v)})
            if not static.get(r.key):  # нет зон — хотя бы ноутбук из модели
                result += [x for x in pre(img, prefix + "L") if x["value"]["rectanglelabels"][0] == "laptop"][:1]
            tasks.append({"data": {"image": ls_url(p), "video": r.key, "t": t, "near_violation": bool(dense)},
                          "predictions": [{"model_version": "draft", "result": result}]})
            n += 1
        total += n
        print(f"{r.key}: {n} кадров" + (f" (метка {tc:g} c)" if tc is not None else ""))
    write_tasks(tasks, ANN / "objects" / "tasks.json")
    print(f"Итого кадров: {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["zones", "objects"])
    ap.add_argument("--zones-export", default=str(ANN / "zones" / "export.json"))
    ap.add_argument("--step", type=float, default=2.0, help="шаг по ролику, c")
    ap.add_argument("--dense-step", type=float, default=0.5, help="шаг около нарушения, c")
    ap.add_argument("--window", type=float, default=3.0, help="± c вокруг метки нарушения")
    ap.add_argument("--dedup", type=float, default=2.5, help="порог «кадр не изменился»")
    a = ap.parse_args()
    {"zones": stage_zones, "objects": stage_objects}[a.stage](a)


if __name__ == "__main__":
    main()
