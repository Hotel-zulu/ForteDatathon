"""Правила нарушений по таблице команды (configs/violations_table.csv).

Каждое правило — наблюдаемый факт на таймлайне модели, без догадок:
  DepartureDetector   — кассир ушёл из кассы; что осталось: деньги / сумка инкассатора / открытый сейф /
                        включённый экран  → cash_left_unsecured / leaving_workplace / safe_left_open /
                        workstation_unlocked                        (X-26, X-15, X-11, X-28)
  CashAtNextClient    — следующий клиент подошёл, а деньги прошлой операции лежат на столе
                                                                    → cash_left_unsecured (X-4, X-38, X-47)
  SafeLeftOpen        — сейф открыт ≥6 c и не закрывается до конца ролика / ≥12 c  → safe_left_open (X-20)
  DoorLeftOpen        — участок двери ≥6 c отличается от закрытого состояния → door_left_open (X-32, X-33)

Пороги подобраны по 42 открытым роликам (scripts/evaluate.py). Время — секунды от начала видео.
"""
from __future__ import annotations

import numpy as np

from src.detectors.base import Candidate, Detector, FrameObs
from src.detectors.scene import Scene, inside_frac, segments


def _box(b, label):
    return [{"xyxy": list(b), "label": label}] if b is not None else []


class DepartureDetector(Detector):
    type_id = "departure"          # сам уход не нарушение — нарушение определяется тем, что осталось
    MIN_EMPTY = 1.9
    LEAVE_POS = 0.45               # таймкод: между «встал с места» и «вышел из комнаты»
    ITEM_FRAC = 0.3
    SCREEN_ON = 0.04

    def finalize(self, tl: list[FrameObs]) -> list[Candidate]:
        if not tl:
            return []
        sc = Scene(tl)
        empty = segments(sc.ts, [c < 0.5 for c in sc.count], 0.0, max_gap=0.15)
        out = []
        for a, b in empty:
            at_end = b >= sc.dur - 0.2
            if a <= sc.ts[0] + 0.5 or (b - a < self.MIN_EMPTY and not (at_end and b - a >= 0.5)):
                continue
            before = [f for f in tl if f.t < a]
            if sum(1 for f in before if sc.room_persons(f)) < 15:   # кассира в комнате не было
                continue
            t_sit = self._last_seated(sc, before, a)
            t_peak = t_sit + self.LEAVE_POS * (a - t_sit)
            # что осталось на столе: с момента ухода и до 5 c после опустения комнаты
            w0, w1 = t_peak, min(sc.dur, a + 5.0)
            ev = [round(t_peak, 2), round(min(sc.dur, a + 1.0), 2)]
            found = []
            if sc.frac("cash", w0, w1) >= self.ITEM_FRAC:
                _, bx = sc.best("cash", w0, w1)
                found.append(("cash_left_unsecured", "на столе остались наличные деньги", _box(bx, "cash")))
            if sc.frac("collector_bag", w0, w1, conf=0.25) >= self.ITEM_FRAC:   # сумка лежит — уверенность ниже
                _, bx = sc.best("collector_bag", w0, w1)
                found.append(("leaving_workplace", "на столе осталась сумка инкассатора — операция не завершена",
                              _box(bx, "collector_bag")))
            if sc.frac("safe_open", w0, w1) >= self.ITEM_FRAC:
                found.append(("safe_left_open", "сейф остался открытым", _box(sc.safe, "safe_open")))
            scr = [f.extra.get("screen") for f in sc.frames(w0, w1) if f.extra.get("screen") is not None]
            if scr and float(np.median(scr)) >= self.SCREEN_ON:
                found.append(("workstation_unlocked",
                              "экран ноутбука остался включённым с открытой рабочей программой (не заблокирован); "
                              "открытая норма требует автоблокировку, обязанность блокировать при уходе — ВНД банка",
                              _box(sc.laptop, "laptop")))
            for tid, what, boxes in found:
                out.append(Candidate(tid, t_sit, a, t_peak=t_peak, confidence=0.7,
                                     observed=f"Около {t_peak:.1f} c кассир покидает рабочее место "
                                              f"(комната пуста с {a:.1f} c); {what}",
                                     boxes=boxes, evidence_ts=ev))
        return out

    @staticmethod
    def _last_seated(sc: Scene, before: list[FrameObs], a: float) -> float:
        """Последний момент, когда последний ушедший человек был на «своём» (сидячем) уровне."""
        last = next((f for f in reversed(before) if sc.room_persons(f)), None)
        if last is None:
            return max(sc.ts[0], a - 3)
        q = max(sc.room_persons(last), key=lambda q: q["conf"])
        tid = q.get("track_id")
        path = [(f.t, p["xyxy"]) for f in before for p in f.persons if tid is not None and p.get("track_id") == tid]
        if len(path) < 5:
            return max(sc.ts[0], a - 3)
        tops = np.array([b[1] for _, b in path])
        seat_top = np.percentile(tops, 80)
        sit = [t for t, b in path if b[1] >= seat_top - 0.03]
        return sit[-1] if sit else path[0][0]


class CashAtNextClient(Detector):
    type_id = "cash_left_unsecured"
    PRE, POST, FRAC = 3.0, 1.0, 0.5

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        out = []
        for k, (a, b) in enumerate(sc.visits()):
            if k == 0:
                continue  # нужен «следующий» клиент
            if sc.frac("cash", a - self.PRE, a + self.POST) >= self.FRAC:
                t, bx = sc.best("cash", a - self.PRE, a + self.POST)
                tp = a + self.POST
                out.append(Candidate(self.type_id, a - self.PRE, b, t_peak=tp, confidence=0.75,
                                     observed=f"Около {tp:.1f} c подходит следующий клиент, а наличные деньги "
                                              f"предыдущей операции лежат на столе, не убраны в средство хранения",
                                     boxes=_box(bx, "cash"), evidence_ts=[round(tp, 2), round(t or tp, 2)]))
        return out


class SafeLeftOpen(Detector):
    type_id = "safe_left_open"
    MIN_OPEN, LONG_OPEN = 5.5, 12.0

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        # разрыв ≥0.3 c = сейф закрывали (в X-20: открыла, прикрыла, снова открыла и ушла за стол)
        runs = segments(sc.ts, [sc.has(f, "safe_open") for f in tl], 0.0, max_gap=0.3)
        out = []
        for a, b in runs:
            ln = b - a
            if ln >= self.LONG_OPEN or (ln >= self.MIN_OPEN and b >= sc.dur - 1.0):
                out.append(Candidate(self.type_id, a, b, t_peak=a, confidence=min(0.9, 0.5 + ln / 30),
                                     observed=f"С {a:.1f} c дверца сейфа открыта и не закрывается "
                                              f"{'до конца записи' if b >= sc.dur - 1.0 else f'{ln:.0f} c'}, "
                                              f"кассир занят другим",
                                     boxes=_box(sc.safe, "safe_open"),
                                     evidence_ts=[round(a, 2), round((a + b) / 2, 2)]))
        return out


class DoorLeftOpen(Detector):
    type_id = "door_left_open"
    THR, MIN_OPEN, SHIFT = 0.6, 6.0, 1.5

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        sig = sc.extra_series("door_sig")
        br = sc.extra_series("bright")
        idx = [i for i, s in enumerate(sig) if s is not None]
        if not idx or sc.door is None:
            return []
        ref_i = next((i for i in idx if not any(inside_frac(q["xyxy"], sc.door) > 0.3
                                                 and q["xyxy"][3] - q["xyxy"][1] > 0.08 for q in tl[i].persons)), idx[0])
        ref, ref_b = np.asarray(sig[ref_i]), br[ref_i] or 1.0
        diff = []
        for i, s in enumerate(sig):
            if s is None or (br[i] and abs(br[i] / ref_b - 1) > 0.25):   # выключили свет и т.п.
                diff.append(0.0)
            else:
                diff.append(float(np.abs(np.asarray(s) - ref).mean()))
        diff = sc._smooth(diff)
        out = []
        for a, b in segments(sc.ts, [d >= self.THR for d in diff], self.MIN_OPEN):
            tp = min(b, a + self.SHIFT)
            out.append(Candidate(self.type_id, a, b, t_peak=tp, confidence=0.7,
                                 observed=f"С {a:.1f} по {b:.1f} c дверь в кассу остаётся открытой",
                                 boxes=_box(sc.door, "door_open"), evidence_ts=[round(tp, 2), round((a + b) / 2, 2)]))
        return out
