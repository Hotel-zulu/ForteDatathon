"""Пиксельные признаки поверх детекций (без нейросетей):

  extra["screen"]   — доля пикселей в рамке ноутбука ярче 90-го перцентиля кадра
                      (светящийся рабочий экран ≈ 0.07–0.2, выключенный/заблокированный ≈ 0–0.02);
  extra["door_sig"] — «отпечаток» участка двери 16×24 (нормированная яркость) для сравнения с закрытым состоянием;
  extra["bright"]   — средняя яркость кадра (чтобы не путать выключенный свет с открытой дверью).
"""
from __future__ import annotations

import cv2
import numpy as np


def _crop(img, b):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
    return img[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]


def door_signature(frame, b) -> np.ndarray:
    g = cv2.cvtColor(_crop(frame, b), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (16, 24)).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


class PixelFeatures:
    """Миксин для Perception: рамки ноутбука и двери фиксируются по первым уверенным детекциям видео."""

    def reset_pixels(self):
        self._lap = self._door = None

    def add_pixels(self, frame, obs):
        obs.extra["bright"] = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        for o in obs.objects:
            if o["cls"] == "laptop" and o["conf"] > 0.6 and self._lap is None:
                self._lap = o["xyxy"]
            if o["cls"].startswith("door") and o["conf"] > 0.6 and self._door is None:
                self._door = o["xyxy"]
        if self._lap is not None:
            v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
            obs.extra["screen"] = float((_crop(v, self._lap) > np.percentile(v, 90)).mean())
        if self._door is not None:
            obs.extra["door_sig"] = door_signature(frame, self._door)
            obs.extra["door_box"] = self._door
