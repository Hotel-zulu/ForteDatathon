"""Сцена по таймлайну: неподвижные объекты, кто в комнате, отрезки времени.

Без ручных зон (на финальном ракурсе их нет): всё опирается на рамки, найденные моделью.
  • «В комнате» = низ рамки человека не выше основания двери/стекла (минус запас) и рамка не крошечная.
    Люди за стеклом и отражения стоят «дальше» стекла → их низ выше основания двери.
"""
from __future__ import annotations

import numpy as np

from src.detectors.base import FrameObs


def med_box(tl: list[FrameObs], names: set[str], conf: float = 0.4):
    bs = [o["xyxy"] for f in tl for o in f.objects if o["cls"] in names and o["conf"] > conf]
    return list(np.median(np.array(bs), 0)) if bs else None


def inside_frac(a, b) -> float:
    """Доля рамки a, лежащая внутри b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + 1e-9)


def segments(ts: list[float], flags: list[bool], min_len: float = 0.0, max_gap: float = 0.6):
    """Склеивает подряд идущие True (с разрывами ≤ max_gap) в отрезки [t0, t1] длиной ≥ min_len."""
    out, st, last = [], None, None
    for t, f in zip(ts, flags):
        if not f:
            continue
        if st is None or t - last > max_gap:
            if st is not None and last - st >= min_len:
                out.append((st, last))
            st = t
        last = t
    if st is not None and last - st >= min_len:
        out.append((st, last))
    return out


class Scene:
    ROOM_MARGIN = 0.07   # насколько низ рамки может быть выше основания двери и всё ещё «в комнате»
    MIN_H = 0.08

    def __init__(self, tl: list[FrameObs]):
        self.tl = tl
        self.ts = [f.t for f in tl]
        self.dur = self.ts[-1] if self.ts else 0.0
        self.door = med_box(tl, {"door_closed", "door_open"})
        self.safe = med_box(tl, {"safe_closed", "safe_open"})
        self.laptop = med_box(tl, {"laptop"})
        self.counter = med_box(tl, {"banknote_counter"})
        raw = [sum(1 for q in f.persons if self.in_room(q)) for f in tl]
        self.count = self._smooth(raw)

    # --- люди ---
    def in_room(self, q: dict) -> bool:
        x1, y1, x2, y2 = q["xyxy"]
        if q.get("conf", 1) < 0.4 or y2 - y1 < self.MIN_H:
            return False
        if self.door is None:
            return y2 - y1 >= 0.12
        return y2 >= self.door[3] - self.ROOM_MARGIN

    def room_persons(self, f: FrameObs) -> list[dict]:
        return [q for q in f.persons if self.in_room(q)]

    def _smooth(self, vals, half: float = 0.5):
        out, j0, j1 = [], 0, 0
        for i, t in enumerate(self.ts):
            while self.ts[j0] < t - half:
                j0 += 1
            while j1 < len(self.ts) and self.ts[j1] <= t + half:
                j1 += 1
            out.append(float(np.median(vals[j0:j1])))
        return out

    # --- объекты ---
    @staticmethod
    def has(f: FrameObs, cls: str, conf: float = 0.4) -> bool:
        return any(o["cls"] == cls and o["conf"] >= conf for o in f.objects)

    def frames(self, t0: float, t1: float) -> list[FrameObs]:
        return [f for f in self.tl if t0 <= f.t <= t1]

    def frac(self, cls: str, t0: float, t1: float, conf: float = 0.4) -> float:
        fs = self.frames(t0, t1)
        return float(np.mean([self.has(f, cls, conf) for f in fs])) if fs else 0.0

    def best(self, cls: str, t0: float, t1: float):
        """(t, рамка) самой уверенной детекции класса в интервале, иначе (None, None)."""
        cand = [(o["conf"], f.t, o["xyxy"]) for f in self.frames(t0, t1) for o in f.objects if o["cls"] == cls]
        if not cand:
            return None, None
        _, t, b = max(cand)
        return t, b

    def extra_series(self, key: str):
        return [f.extra.get(key) for f in self.tl]

    def visits(self, min_len: float = 3.0):
        """Отрезки, когда в комнате ≥2 человек (кассир + клиент/инкассатор)."""
        return segments(self.ts, [c >= 2 for c in self.count], min_len)
