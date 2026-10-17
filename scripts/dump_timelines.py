"""Прогнать восприятие (модели) по всем видео и сохранить покадровые наблюдения в workdir/timelines/<видео>.json.
Нужно, чтобы подбирать правила нарушений по данным, не пересчитывая модели каждый раз.

    python scripts/dump_timelines.py               # все видео, 10 к/с
    python scripts/dump_timelines.py --only X-26 --fps 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.labels import norm_name  # noqa: E402
from src.video_io import iter_frames, probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--fps", type=float, default=config.SAMPLE_FPS)
    a = ap.parse_args()
    from src.detectors.cashier_perception import CashierPerception
    per = CashierPerception()
    out = config.WORKDIR / "timelines"
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(config.VIDEOS_DIR.glob("*.mov"), key=lambda p: int(norm_name(p.stem).split("-")[1]))
    if a.only:
        keep = {norm_name(x) for x in a.only}
        vids = [v for v in vids if norm_name(v.stem) in keep]
    for v in vids:
        t0 = time.time()
        per.reset()
        m = probe(str(v))
        tl = []
        for t, f in iter_frames(str(v), a.fps, config.MAX_SIDE):
            o = asdict(per(t, f))
            o["extra"] = {k: (np.round(x, 3).tolist() if isinstance(x, np.ndarray) else x) for k, x in o["extra"].items()}
            tl.append(o)
        (out / f"{norm_name(v.stem)}.json").write_text(json.dumps(
            {"video": norm_name(v.stem), "duration": m.duration_sec, "fps": a.fps, "frames": tl}))
        print(f"{norm_name(v.stem):6s} {len(tl):5d} кадров  {time.time() - t0:5.1f} c", flush=True)


if __name__ == "__main__":
    main()
