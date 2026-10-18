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
            if sc.frac("key_in_lock", w0, w1, conf=0.35) >= self.ITEM_FRAC:
                _, bx = sc.best("key_in_lock", w0, w1)
                found.append(("key_left_in_lock", "в замке сейфа остался ключ", _box(bx, "key_in_lock")))
            if sc.frac("documents", w0, w1, conf=0.35) >= self.ITEM_FRAC:
                _, bx = sc.best("documents", w0, w1)
                found.append(("documents_left_open", "на столе остались документы", _box(bx, "documents")))
            # экран: класс модели screen_on (если есть) или пиксельный признак яркости
            scr = [f.extra.get("screen") for f in sc.frames(w0, w1) if f.extra.get("screen") is not None]
            scr_on = sc.frac("screen_on", w0, w1, conf=0.4)
            scr_lock = sc.frac("screen_locked", w0, w1, conf=0.4)
            bright = bool(scr) and float(np.median(scr)) >= self.SCREEN_ON
            # нужны оба признака: модель видит рабочий экран И экран реально светится (экран блокировки тёмный)
            if (scr_on >= self.ITEM_FRAC and scr_on > scr_lock and bright) or (not scr_on and not scr_lock and bright):
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
        """Последний момент, когда уходящий человек был на «своём» (сидячем) уровне.
        Путь человека восстанавливается назад по кадрам по ближайшей рамке — не зависит от ID трекера."""
        idx = next((i for i in range(len(before) - 1, -1, -1) if sc.room_persons(before[i])), None)
        if idx is None:
            return max(sc.ts[0], a - 3)
        cur = max(sc.room_persons(before[idx]), key=lambda q: q["conf"])["xyxy"]
        path, miss = [(before[idx].t, cur)], 0
        for f in reversed(before[:idx]):
            ps = sc.room_persons(f)
            if not ps:
                miss += 1
                if miss > 10:
                    break
                continue
            c = ((cur[0] + cur[2]) / 2, (cur[1] + cur[3]) / 2)
            q = min(ps, key=lambda q: (((q["xyxy"][0] + q["xyxy"][2]) / 2 - c[0]) ** 2 +
                                       ((q["xyxy"][1] + q["xyxy"][3]) / 2 - c[1]) ** 2))
            qc = ((q["xyxy"][0] + q["xyxy"][2]) / 2, (q["xyxy"][1] + q["xyxy"][3]) / 2)
            if ((qc[0] - c[0]) ** 2 + (qc[1] - c[1]) ** 2) ** 0.5 > 0.12 * sc.scale:
                miss += 1
                if miss > 10:
                    break
                continue
            miss, cur = 0, q["xyxy"]
            path.append((f.t, cur))
        path.reverse()
        if len(path) < 5:
            return max(sc.ts[0], a - 3)
        tops = np.array([b[1] for _, b in path])
        seat_top = np.percentile(tops, 80)
        sit = [t for t, b in path if b[1] >= seat_top - 0.03 * sc.scale]
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
    THR, MIN_OPEN, SHIFT, MIN_OPEN_DET = 0.6, 6.0, 1.5, 9.0

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
        # 1) модель видит открытую дверь (класс door_open): обычный проход 2–4 c, нарушение — ≥9 c
        for a, b in segments(sc.ts, [sc.has(f, "door_open", 0.5) for f in tl], self.MIN_OPEN_DET, max_gap=0.5):
            tp = min(b, a + self.SHIFT)
            out.append(Candidate(self.type_id, a, b, t_peak=tp, confidence=0.8,
                                 observed=f"С {a:.1f} по {b:.1f} c дверь в кассу остаётся открытой",
                                 boxes=_box(sc.best("door_open", a, b)[1], "door_open"),
                                 evidence_ts=[round(tp, 2), round((a + b) / 2, 2)]))
        # 2) запасной признак без модели: участок двери отличается от закрытого состояния
        for a, b in segments(sc.ts, [d >= self.THR for d in diff], self.MIN_OPEN):
            tp = min(b, a + self.SHIFT)
            out.append(Candidate(self.type_id, a, b, t_peak=tp, confidence=0.7,
                                 observed=f"С {a:.1f} по {b:.1f} c дверь в кассу остаётся открытой",
                                 boxes=_box(sc.door, "door_open"), evidence_ts=[round(tp, 2), round((a + b) / 2, 2)]))
        return out


class NoIdentification(Detector):
    """Инкассатор с сумкой обслужен, а удостоверение за все его визиты так и не появилось.
    Только визиты инкассатора: у обычных клиентов на части ракурсов удостоверение не видно, а инкассатор
    предъявляет удостоверение/доверенность всегда (X-16 — предъявил, X-17 — нет). Визиты одного инкассатора
    считаются вместе (трекер может разрезать визит на части)."""
    type_id = "no_client_identification"
    MIN_VISIT, ID_CONF, BAG_CONF, MIN_BAG, AT = 4.0, 0.35, 0.5, 20, 5.0

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        coll, bag_total, id_total = [], 0, 0
        for a, b in sc.visits(self.MIN_VISIT):
            fs = sc.frames(a - 2, b + 2)
            bag = sum(1 for f in fs if sc.has(f, "collector_bag", self.BAG_CONF))
            if bag >= self.MIN_BAG:
                coll.append((a, b))
                bag_total += bag
                id_total += sum(1 for f in fs if sc.has(f, "id_card", self.ID_CONF))
        if not coll or id_total >= max(10, 0.05 * bag_total):
            return []
        a, b = coll[0]
        tp = a + min(self.AT, (b - a) / 2)
        return [Candidate(self.type_id, a, coll[-1][1], t_peak=tp, confidence=0.6,
                          observed=f"С {a:.1f} c у кассы инкассатор с сумкой, операция идёт, но удостоверение "
                                   f"личности кассиру не предъявляется",
                          boxes=_box(sc.best("collector_bag", a, b)[1], "collector_bag"),
                          evidence_ts=[round(tp, 2), round((a + b) / 2, 2)])]

# ======================= правила на новых классах (разметка 18.10) =======================

def _counting(sc: Scene, f: FrameObs) -> bool:
    """Признак пересчёта на оборудовании: класс cash_in_counter или деньги на/в счётчике."""
    if sc.has(f, "cash_in_counter", 0.35):
        return True
    if sc.counter is None:
        return False
    return any(o["cls"] == "cash" and o["conf"] >= 0.4 and inside_frac(o["xyxy"], sc.counter) > 0.3 for o in f.objects)


class PersonalItems(Detector):
    """Личная вещь (сумка, одежда) лежит на рабочем столе кассира ≥3 c."""
    type_id = "personal_items_in_cash_area"
    CONF, MIN_LEN = 0.45, 3.0

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        out = []
        for a, b in segments(sc.ts, [sc.has(f, "personal_item", self.CONF) for f in tl], self.MIN_LEN, 0.6)[:1]:
            tp = a + 1.0
            out.append(Candidate(self.type_id, a, b, t_peak=tp, confidence=0.65,
                                 observed=f"С {a:.1f} c на рабочем месте кассира лежит личная вещь (сумка/одежда)",
                                 boxes=_box(sc.best("personal_item", a, b)[1], "personal_item"),
                                 evidence_ts=[round(tp, 2), round((a + b) / 2, 2)]))
        return out


class ThirdPerson(Detector):
    """Во время обслуживания у кассы трое и больше людей, и в этот момент передаётся документ/деньги."""
    type_id = "third_party_document"
    MIN_LEN = 3.0

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        out = []
        for a, b in segments(sc.ts, [c >= 3 for c in sc.count], self.MIN_LEN):
            fs = sc.frames(a, b)
            ids = sum(1 for f in fs if sc.has(f, "id_card", 0.35))
            if ids >= 3:
                tp = a + 1.0
                out.append(Candidate(self.type_id, a, b, t_peak=tp, confidence=0.6,
                                     observed=f"С {a:.1f} c у кассы находятся три человека; удостоверение личности "
                                              f"предъявляет/передаёт не тот человек, который обслуживается",
                                     boxes=_box(sc.best("id_card", a, b)[1], "id_card"),
                                     evidence_ts=[round(tp, 2), round((a + b) / 2, 2)]))
        return out


class NotCountedToSafe(Detector):
    """Деньги убраны в сейф, а за 20 c до этого не было пересчёта на счётчике."""
    type_id = "cash_not_counted"
    LOOKBACK = 20.0

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        out = []
        for a, b in segments(sc.ts, [sc.has(f, "safe_open") for f in tl], 1.0, 0.3):
            pre = sc.frames(a - self.LOOKBACK, a)
            cash = sum(1 for f in pre if sc.has(f, "cash"))
            counted = sum(1 for f in pre if _counting(sc, f))
            if cash >= 10 and counted == 0:
                out.append(Candidate(self.type_id, a - self.LOOKBACK, b, t_peak=a, confidence=0.55,
                                     observed=f"Около {a:.1f} c наличные убираются в сейф; перед этим пересчёт "
                                              f"на счётчике банкнот не проводился",
                                     boxes=_box(sc.safe, "safe_open"), evidence_ts=[round(a, 2), round(max(0, a - 5), 2)]))
        return out


class NoMachineCheck(Detector):
    """Кассир долго работает с наличными, но ни разу не использует счётчик-детектор банкнот."""
    type_id = "no_authenticity_check"
    MIN_CASH = 5.0

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        if any(_counting(sc, f) for f in tl):
            return []
        runs = segments(sc.ts, [sc.has(f, "cash") for f in tl], 1.0, 0.6)
        total = sum(b - a for a, b in runs)
        if total < self.MIN_CASH:
            return []
        a, b = max(runs, key=lambda r: r[1] - r[0])
        tp = (a + b) / 2
        return [Candidate(self.type_id, a, b, t_peak=tp, confidence=0.5,
                          observed=f"С {a:.1f} по {b:.1f} c кассир работает с наличными вручную; счётчик банкнот "
                                   f"(проверка подлинности на оборудовании) не используется",
                          boxes=_box(sc.best("cash", a, b)[1], "cash"), evidence_ts=[round(tp, 2)])]


class ClientLeftEarly(Detector):
    """Клиент уходит (в комнате остаётся один), а деньги операции ещё лежат на столе/в счётчике."""
    type_id = "client_left_before_completion"
    NEED_COUNTER = False   # требовать, чтобы деньги были в счётчике (строже, меньше ложных)

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        out = []
        for a, b in sc.visits(4.0):
            if b >= sc.dur - 1.0:
                continue   # клиент не ушёл до конца записи
            around = sc.frac("cash", b - 1.0, b + 2.0)
            in_ctr = any(_counting(sc, f) for f in sc.frames(b - 3.0, b + 1.0))
            if around >= 0.5 and (in_ctr or not self.NEED_COUNTER):
                out.append(Candidate(self.type_id, a, b, t_peak=b, confidence=0.5,
                                     observed=f"Около {b:.1f} c клиент уходит от кассы, а наличные операции ещё "
                                              f"на столе/в счётчике — операция не завершена",
                                     boxes=_box(sc.best("cash", b - 1, b + 2)[1], "cash"),
                                     evidence_ts=[round(b, 2), round(max(a, b - 2), 2)]))
        return out


class CashLeftAfterHandover(Detector):
    """Выдача прошла (клиент у кассы), а остаток наличных так и лежит на столе до конца записи / ухода клиента,
    не убран в средство хранения и не в счётчике (X-1)."""
    type_id = "cash_left_unsecured"
    MIN_LEN, TAIL = 4.0, 1.0

    def finalize(self, tl):
        if not tl:
            return []
        sc = Scene(tl)
        out = []
        for a, b in sc.visits(5.0):
            # непрерывный отрезок «деньги на столе», доходящий до конца визита
            runs = [r for r in segments(sc.ts, [sc.has(f, "cash") and not _counting(sc, f) for f in tl], self.MIN_LEN, 0.6)
                    if r[1] >= b - self.TAIL and r[0] > a + 3.0]
            if not runs:
                continue
            r0, r1 = runs[-1]
            if any(sc.has(f, "safe_open") for f in sc.frames(r0, r1)):
                continue   # кассир в это время убирал деньги в сейф
            tp = r0 + 1.0
            out.append(Candidate(self.type_id, r0, r1, t_peak=tp, confidence=0.55,
                                 observed=f"С {r0:.1f} c после выдачи остаток наличных лежит на столе и до "
                                          f"{r1:.1f} c не убран в средство хранения",
                                 boxes=_box(sc.best("cash", r0, r1)[1], "cash"),
                                 evidence_ts=[round(tp, 2), round((r0 + r1) / 2, 2)]))
        return out
