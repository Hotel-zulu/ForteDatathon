"""Быстрая проверка правил на сохранённых таймлайнах (без пересчёта моделей, секунды вместо 25 минут).

    python scripts/dump_timelines.py          # один раз после смены модели
    python scripts/eval_timelines.py          # после каждой правки правил

Сравнивает с configs/labels.csv (вердикт, таймкод) и configs/violations_table.csv (тип по таблице команды).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import rules  # noqa: E402
from src.config import WORKDIR  # noqa: E402
from src.detectors import build_detectors  # noqa: E402
from src.detectors.base import FrameObs  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.pipeline import _dedup  # noqa: E402


def load_tl(p: Path) -> list[FrameObs]:
    d = json.loads(p.read_text())
    return [FrameObs(t=f["t"], persons=f["persons"], objects=f["objects"], extra=f.get("extra", {})) for f in d["frames"]]


def main():
    types = rules.load_types()
    team = {r["video"]: r for r in csv.DictReader(open(ROOT / "configs" / "violations_table.csv", encoding="utf-8"))}
    want = {}
    for tid, vt in types.items():
        pass
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "regulations.yaml").read_text(encoding="utf-8"))
    for tid, v in cfg["violation_types"].items():
        for vid in v.get("videos", []):
            want.setdefault(vid, set()).add(tid)
    labels = load_labels()
    tp = tn = fp = fn = type_ok = 0
    errs, rows = [], []
    for r in labels.itertuples():
        p = WORKDIR / "timelines" / f"{r.key}.json"
        if not p.exists():
            continue
        tl = load_tl(p)
        cands = [c for d in build_detectors() for c in d.finalize(tl) if types.get(c.type_id) and types[c.type_id].enabled]
        cands = _dedup(cands)
        pred = int(bool(cands))
        msg = ""
        if pred and r.violation:
            tp += 1
            e = min(abs(c.t_peak - r.timecode_sec) for c in cands)
            errs.append(e)
            ok = any(c.type_id in want.get(r.key, set()) for c in cands)
            type_ok += ok
            msg = f"✅ tc_err={e:.1f}  тип {'совпал' if ok else '≠ таблице: ' + ','.join(want.get(r.key, []))}"
        elif not pred and not r.violation:
            tn += 1; msg = "✅"
        elif pred:
            fp += 1; msg = "⚠ ЛОЖНОЕ"
        else:
            fn += 1; msg = f"❌ пропуск ({','.join(want.get(r.key, ['?']))})"
        found = "; ".join(f"{c.type_id}@{c.t_peak:.1f}" for c in cands)
        rows.append({"video": r.key, "label": int(r.violation), "pred": pred,
                     "gt_tc": float(r.timecode_sec) if r.violation else None,
                     "found": [{"type": c.type_id, "name": types[c.type_id].name, "t": round(c.t_peak, 2)} for c in cands],
                     "tc_err": round(min(abs(c.t_peak - r.timecode_sec) for c in cands), 2) if (pred and r.violation) else None,
                     "team_types": sorted(want.get(r.key, [])),
                     "team_text": team.get(r.key, {}).get("violation", "")})
        print(f"{r.key:5s} gt={'%.0f' % r.timecode_sec if r.violation else '-':>3s}  {msg:40s} {found}")
    n = tp + tn + fp + fn
    print(f"\nAccuracy вердикта {(tp + tn) / n:.3f} ({n} видео) | нашли {tp} из {tp + fn} нарушений | ложных {fp} из {fp + tn}")
    if errs:
        import numpy as np
        print(f"Таймкод: медиана ошибки {np.median(errs):.1f} c, ≤2 c: {np.mean(np.array(errs) <= 2):.0%} | тип совпал {type_ok}/{tp}")
    import time, os
    out = WORKDIR / "eval"; out.mkdir(parents=True, exist_ok=True)
    det = os.getenv("DETECTOR_WEIGHTS", "models/detector/weights/best.pt")
    summary = {"available": True, "created_at": time.time(), "videos": n, "accuracy": round((tp + tn) / n, 3),
               "tp": tp, "tn": tn, "fp": fp, "fn": fn, "type_ok": type_ok,
               "tc_median": round(float(__import__('numpy').median(errs)), 2) if errs else None,
               "tc_within2": round(float(__import__('numpy').mean(__import__('numpy').array(errs) <= 2)), 2) if errs else None,
               "detector": det, "rows": rows}
    (out / "latest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Сохранено: {out / 'latest.json'}")


if __name__ == "__main__":
    main()
