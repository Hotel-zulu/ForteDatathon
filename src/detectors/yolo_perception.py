"""Perception на YOLO (Ultralytics): люди + предметы + трекинг ByteTrack.

⚠ ЛИЦЕНЗИЯ: ultralytics и веса YOLO распространяются под AGPL-3.0. По FAQ п.24 разрешены только
open-source с лицензией для коммерческого использования — для ИССЛЕДОВАНИЯ и разметки годится,
для финальной сдачи замените на Apache-2.0 модель (RT-DETR/D-FINE через transformers) или
сознательно примите риск. Интерфейс тот же (Perception), поэтому замена — один файл.

Устройство: Apple Silicon -> "mps", NVIDIA -> "cuda", иначе "cpu".
"""
from __future__ import annotations

import os

import numpy as np

from src.config import MODELS_DIR
from src.detectors.base import FrameObs, Perception

# COCO-классы, полезные для кассы. Деньги/сумки инкассаторов/ящики в COCO нет —
# их придётся дообучать (разметка на кадрах из scripts/extract_frames.py).
COCO_KEEP = {0: "person", 67: "phone", 24: "backpack", 26: "handbag", 28: "suitcase",
             63: "laptop", 73: "book", 41: "cup", 39: "bottle", 76: "scissors"}


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloPerception(Perception):
    def __init__(self, weights: str | None = None, conf: float = 0.25, imgsz: int = 960, track: bool = True):
        from ultralytics import YOLO
        weights = weights or os.getenv("YOLO_WEIGHTS", str(MODELS_DIR / "yolo11s.pt"))
        self.model = YOLO(weights)
        self.device = os.getenv("DEVICE") or pick_device()
        self.conf, self.imgsz, self.track = conf, imgsz, track

    def __call__(self, t: float, frame: np.ndarray) -> FrameObs:
        kw = dict(conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
                  classes=list(COCO_KEEP))
        r = (self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kw) if self.track
             else self.model.predict(frame, **kw))[0]
        h, w = frame.shape[:2]
        obs = FrameObs(t=t)
        if r.boxes is None:
            return obs
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for xyxy, c, p, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), ids):
            box = [xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h]
            name = COCO_KEEP.get(c, str(c))
            if name == "person":
                # роль пока неизвестна: определяется зонами (TODO, см. docs/ARCHITECTURE.md)
                obs.persons.append({"track_id": tid, "xyxy": box, "conf": p, "role": "unknown"})
            else:
                obs.objects.append({"cls": name, "track_id": tid, "xyxy": box, "conf": p})
        return obs

    def reset(self):
        # сброс трекера между видео
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor.trackers = None
