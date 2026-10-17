"""Чтение видео: метаданные и прореживание кадров по РЕАЛЬНОМУ времени (точные таймкоды)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int
    duration_sec: float


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta = VideoMeta(str(path), float(fps), n, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), n / fps if fps else 0.0)
    cap.release()
    return meta


def resize_max_side(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side:
        return frame
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return frame if s >= 1 else cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def iter_frames(path: str | Path, target_fps: float = 10.0, max_side: int | None = None,
                start_sec: float = 0.0, end_sec: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Отдаёт (timestamp_sec, кадр BGR) примерно target_fps раз в секунду.

    Кадры выбираются по времени, а не «каждый N-й», поэтому работает при 29.97 / 59.94 fps
    и таймкоды не уплывают. grab() без декодирования пропускаемых кадров — быстро.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    step = 1.0 / target_fps
    next_t, i = start_sec, int(start_sec * fps)
    try:
        while True:
            if not cap.grab():
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t = pos if pos > 0 else i / fps
            i += 1
            if end_sec is not None and t > end_sec:
                break
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            next_t += step
            yield round(t, 3), resize_max_side(frame, max_side)
    finally:
        cap.release()


def frame_at(path: str | Path, t_sec: float, max_side: int | None = None) -> np.ndarray | None:
    """Точный кадр по таймкоду (для кадров-доказательств — из оригинала, без прореживания)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(t_sec, 0) * 1000)
    ok, frame = cap.read()
    cap.release()
    return resize_max_side(frame, max_side) if ok else None
