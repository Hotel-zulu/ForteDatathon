"""Прогнать восприятие (модели) по всем видео и сохранить покадровые наблюдения в workdir/timelines/<видео>.json.
Нужно, чтобы подбирать правила нарушений по данным, не пересчитывая модели каждый раз.

    python scripts/dump_timelines.py               # все видео, 10 к/с
    python scripts/dump_timelines.py --only X-26 --fps 5
    python scripts/dump_timelines.py --aug persp --out timelines_persp   # стресс-тест «другой ракурс»

--aug: искусственно меняет ракурс каждого кадра (проверка устойчивости, модель этих видео так не видела):
  mirror — зеркально; zoom — камера ближе (центр 75%); far — камера дальше (кадр 70% + поля);
  persp — камера сбоку (перспектива); rot — наклон камеры 12°; dark — темнее и шумнее.
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


def augment(img, kind):
    import cv2
    h, w = img.shape[:2]
    if kind == "mirror":
        return img[:, ::-1].copy()
    if kind == "zoom":
        y0, x0 = int(h * 0.125), int(w * 0.125)
        return cv2.resize(img[y0:h - y0, x0:w - x0], (w, h))
    if kind == "far":
        small = cv2.resize(img, (int(w * 0.7), int(h * 0.7)))
        out = np.full_like(img, 110)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        return out
    if kind == "persp":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * 0.12, h * 0.06], [w * 0.95, 0], [w, h], [0, h * 0.94]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(110, 110, 110))
    if kind == "rot":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.1)
        return cv2.warpAffine(img, M, (w, h), borderValue=(110, 110, 110))
    if kind == "dark":
        noisy = img.astype(np.float32) * 0.6 + np.random.normal(0, 8, img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--fps", type=float, default=config.SAMPLE_FPS)
    ap.add_argument("--aug", choices=["mirror", "zoom", "far", "persp", "rot", "dark"])
    ap.add_argument("--out", default="timelines", help="папка в workdir/")
    a = ap.parse_args()
    from src.detectors import build_perception
    per = build_perception()
    out = config.WORKDIR / a.out
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
            o = asdict(per(t, augment(f, a.aug) if a.aug else f))
            o["extra"] = {k: (np.round(x, 3).tolist() if isinstance(x, np.ndarray) else x) for k, x in o["extra"].items()}
            tl.append(o)
        (out / f"{norm_name(v.stem)}.json").write_text(json.dumps(
            {"video": norm_name(v.stem), "duration": m.duration_sec, "fps": a.fps, "frames": tl}))
        print(f"{norm_name(v.stem):6s} {len(tl):5d} кадров  {time.time() - t0:5.1f} c", flush=True)


if __name__ == "__main__":
    main()
