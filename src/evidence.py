"""Кадры-доказательства: точный кадр из оригинала, разметка, размытие лиц, JPEG -> base64.

Лица только РАЗМЫВАЮТСЯ (детектор Haar из OpenCV), а не распознаются — идентификация запрещена ТЗ.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from src.video_io import frame_at

_face = None


def _face_detector():
    """Haar-каскад из OpenCV 4.x. В OpenCV 5 его нет в основном пакете — тогда размытие пропускается
    (TODO: заменить на детектор голов/людей из perception — сверху лица и так видны плохо)."""
    global _face
    if _face is None:
        try:
            _face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except AttributeError:
            _face = False
    return _face


def blur_faces(img: np.ndarray) -> np.ndarray:
    det = _face_detector()
    if not det:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in det.detectMultiScale(gray, 1.1, 5, minSize=(24, 24)):
        roi = img[y:y + h, x:x + w]
        img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(w, h) / 4)
    return img


def annotate(img: np.ndarray, boxes: list[dict] | None, caption: str | None) -> np.ndarray:
    """boxes: [{"xyxy": [x1,y1,x2,y2] в долях 0..1, "label": "..."}]"""
    h, w = img.shape[:2]
    for b in boxes or []:
        x1, y1, x2, y2 = (np.array(b["xyxy"]) * [w, h, w, h]).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        if b.get("label"):
            cv2.putText(img, b["label"], (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if caption:
        cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(img, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def to_base64_jpeg(img: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Не удалось закодировать JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def make_evidence(video: str | Path, timestamps: list[float], boxes: list[dict] | None = None,
                  blur: bool = True, max_side: int = 1280) -> list[dict]:
    out = []
    for t in timestamps[:3]:  # по ТЗ 1–3 кадра
        img = frame_at(video, t, max_side=max_side)
        if img is None:
            continue
        if blur:
            img = blur_faces(img)
        img = annotate(img, boxes, f"t = {t:.1f} s")
        out.append({"timestamp_sec": round(float(t), 2), "image_base64": to_base64_jpeg(img)})
    return out
