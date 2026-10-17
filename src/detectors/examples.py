"""Примеры rule-детекторов: показывают, как из таймлайна наблюдений получить кандидатов.

Они заработают, когда perception начнёт заполнять persons/objects. Пороговые значения —
стартовые, их нужно подобрать на open/videos по configs/labels.csv (scripts/evaluate.py).
"""
from __future__ import annotations

import numpy as np

from src.detectors.base import Candidate, Detector, FrameObs, Perception


class NullPerception(Perception):
    """Заглушка: ничего не видит. Позволяет запустить весь сервис end-to-end до готовности моделей."""

    def __call__(self, t: float, frame: np.ndarray) -> FrameObs:
        return FrameObs(t=t)


def _segments(flags: list[tuple[float, bool]], min_len: float, max_gap: float = 0.5):
    """Склеивает подряд идущие True в отрезки [t0, t1] длиннее min_len."""
    segs, start, last = [], None, None
    for t, f in flags:
        if f:
            if start is None or (last is not None and t - last > max_gap):
                if start is not None and last - start >= min_len:
                    segs.append((start, last))
                start = t
            last = t
    if start is not None and last - start >= min_len:
        segs.append((start, last))
    return segs


class LeavingWorkplaceDetector(Detector):
    """Кассир отсутствует на рабочем месте, пока у окна есть клиент (п. 22 Правил №231)."""
    type_id = "leaving_workplace"
    MIN_ABSENCE_SEC = 3.0

    def finalize(self, timeline: list[FrameObs]) -> list[Candidate]:
        flags = []
        for o in timeline:
            roles = {p.get("role") for p in o.persons}
            flags.append((o.t, "client" in roles and "cashier" not in roles))
        out = []
        for t0, t1 in _segments(flags, self.MIN_ABSENCE_SEC):
            out.append(Candidate(self.type_id, t0, t1, t_peak=t0, confidence=min(1.0, (t1 - t0) / 10),
                                 observed=f"С {t0:.1f} по {t1:.1f} c у окна находится клиент, кассира на рабочем месте нет",
                                 evidence_ts=[t0, (t0 + t1) / 2]))
        return out


class PhoneDetector(Detector):
    """Смартфон в руках кассира в зоне кассы (п. 12 Правил №231)."""
    type_id = "phone_in_cash_area"
    MIN_SEC, MIN_CONF = 1.0, 0.5

    def finalize(self, timeline: list[FrameObs]) -> list[Candidate]:
        flags, best = [], {}
        for o in timeline:
            phones = [x for x in o.objects if x.get("cls") == "phone" and x.get("conf", 0) >= self.MIN_CONF]
            flags.append((o.t, bool(phones)))
            if phones:
                best[o.t] = max(phones, key=lambda x: x["conf"])
        out = []
        for t0, t1 in _segments(flags, self.MIN_SEC):
            ts = [t for t in best if t0 <= t <= t1]
            peak = max(ts, key=lambda t: best[t]["conf"])
            out.append(Candidate(self.type_id, t0, t1, t_peak=peak, confidence=best[peak]["conf"],
                                 observed=f"С {t0:.1f} по {t1:.1f} c в зоне кассы виден мобильный телефон",
                                 boxes=[{"xyxy": best[peak]["xyxy"], "label": "phone"}],
                                 evidence_ts=[peak]))
        return out
