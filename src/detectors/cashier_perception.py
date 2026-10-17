"""Perception для кассы: НАША обученная модель (деньги, удостоверение, телефон, счётчик, ноутбук,
сейф, дверь, сумки…) + люди из COCO-модели с трекингом + два простых пиксельных признака:

  extra["screen"]   — доля пикселей в рамке ноутбука ярче 90-го перцентиля кадра
                      (светящийся рабочий экран ≈ 0.07–0.2, выключенный/заблокированный ≈ 0–0.02);
  extra["door_sig"] — «отпечаток» участка двери 16×24 (нормированная яркость) для сравнения с закрытым
                      состоянием; extra["bright"] — средняя яркость кадра (чтобы не путать выключенный свет).

Координаты — в долях кадра (0..1). Роли и фильтр отражений — в src/detectors/scene.py.
⚠ ultralytics и веса — AGPL-3.0 (см. yolo_perception.py). Интерфейс тот же, замена — один файл.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from src.config import MODELS_DIR
from src.detectors.base import FrameObs, Perception
from src.detectors.yolo_perception import pick_device


def _crop(frame, b):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
    return frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]


def door_signature(frame, b) -> np.ndarray:
    g = cv2.cvtColor(_crop(frame, b), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (16, 24)).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


class CashierPerception(Perception):
    def __init__(self, conf: float = 0.25, imgsz: int = 960):
        from ultralytics import YOLO
        self.obj = YOLO(os.getenv("DETECTOR_WEIGHTS", str(MODELS_DIR / "detector" / "weights" / "best.pt")))
        self.ppl = YOLO(os.getenv("YOLO_WEIGHTS", str(MODELS_DIR / "yolo11s.pt")))
        self.device = os.getenv("DEVICE") or pick_device()
        self.conf, self.imgsz = conf, imgsz
        self.names = self.obj.names
        self.reset()

    def __call__(self, t: float, frame: np.ndarray) -> FrameObs:
        h, w = frame.shape[:2]
        obs = FrameObs(t=t)
        r = self.obj.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            obs.objects.append({"cls": self.names[c], "xyxy": [b[0] / w, b[1] / h, b[2] / w, b[3] / h],
                                "conf": round(p, 3)})
        r = self.ppl.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, imgsz=self.imgsz,
                           device=self.device, verbose=False, classes=[0])[0]
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for b, p, tid in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids):
            obs.persons.append({"track_id": tid, "xyxy": [b[0] / w, b[1] / h, b[2] / w, b[3] / h],
                                "conf": round(p, 3), "role": "unknown"})
        self._pixels(frame, obs)
        return obs

    def _pixels(self, frame, obs):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        obs.extra["bright"] = float(gray.mean())
        # неподвижные объекты: фиксируем рамку по первым уверенным детекциям
        for o in obs.objects:
            if o["cls"] == "laptop" and o["conf"] > 0.6 and self._lap is None:
                self._lap = o["xyxy"]
            if o["cls"].startswith("door") and o["conf"] > 0.6 and self._door is None:
                self._door = o["xyxy"]
        if self._lap is not None:
            hsv_v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
            thr = np.percentile(hsv_v, 90)
            obs.extra["screen"] = float((_crop(hsv_v, self._lap) > thr).mean())
        if self._door is not None:
            obs.extra["door_sig"] = door_signature(frame, self._door)
            obs.extra["door_box"] = self._door

    def reset(self):
        self._lap = self._door = None
        p = getattr(self.ppl, "predictor", None)
        if p is not None and hasattr(p, "trackers"):
            del p.trackers
