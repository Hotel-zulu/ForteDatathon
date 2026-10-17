"""Локальная оценка на открытых видео — та же логика, что у жюри (критерии 1 и 2).

    python scripts/evaluate.py                 # все видео
    python scripts/evaluate.py --only X-1 X-3  # выборочно

Печатает по каждому видео: метка / вердикт / ошибка таймкода; итог: accuracy вердикта,
пропуски (FN — недопустимы по FAQ 16), ложные срабатывания (FP), средняя ошибка таймкода.
Результаты: workdir/eval/<время>.csv — сравнивайте эксперименты.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import VIDEOS_DIR, WORKDIR  # noqa: E402
from src.labels import load_labels, norm_name  # noqa: E402
from src.pipeline import analyze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tol", type=float, default=2.0, help="допуск по таймкоду, c (для статистики)")
    a = ap.parse_args()

    labels = load_labels()
    if a.only:
        keep = {norm_name(x) for x in a.only}
        labels = labels[labels.key.isin(keep)]
    rows = []
    for r in labels.itertuples():
        if not isinstance(r.file, str):
            print(f"⚠ нет файла для {r.name}")
            continue
        t0 = time.time()
        res = analyze(VIDEOS_DIR / r.file, with_frames=False)
        pred = int(res["verdict"] == "violation")
        tcs = [v["timecode_sec"] for v in res["violations"]]
        err = min(abs(t - r.timecode_sec) for t in tcs) if (pred and r.violation and tcs) else None
        rows.append(dict(key=r.key, label=r.violation, pred=pred, gt_tc=r.timecode_sec,
                         pred_tc=tcs[0] if tcs else None, tc_err=err, n_viol=len(tcs),
                         types="; ".join(v["type"] for v in res["violations"]), sec=round(time.time() - t0, 1)))
        mark = "✅" if pred == r.violation else ("❌ ПРОПУСК" if r.violation else "⚠ FP")
        print(f"{r.key:6s} label={r.violation} pred={pred} {mark}  tc_err={err}  ({rows[-1]['sec']} c)")

    df = pd.DataFrame(rows)
    if df.empty:
        return
    fn = int(((df.label == 1) & (df.pred == 0)).sum())
    fp = int(((df.label == 0) & (df.pred == 1)).sum())
    tc = df.tc_err.dropna()
    print("\n=== ИТОГ ===")
    print(f"Accuracy вердикта: {(df.label == df.pred).mean():.3f}  ({len(df)} видео)")
    print(f"Пропуски (FN): {fn} из {int(df.label.sum())}   Ложные (FP): {fp} из {int((1 - df.label).sum())}")
    if len(tc):
        print(f"Таймкод: средняя ошибка {tc.mean():.2f} c, медиана {tc.median():.2f} c, в пределах ±{a.tol} c: {(tc <= a.tol).mean():.0%}")
    print(f"Макс. время на видео: {df.sec.max()} c (лимит жюри 300 c)")
    out = WORKDIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"eval_{time.strftime('%m%d_%H%M')}.csv"
    df.to_csv(p, index=False)
    print(f"Сохранено: {p}")


if __name__ == "__main__":
    main()
