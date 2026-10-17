"""Нарезка кадров из видео (для разметки, просмотра, обучения детектора).

    python scripts/extract_frames.py                       # все видео из open/videos, 10 fps
    python scripts/extract_frames.py --fps 2               # обзорно, для просмотра командой
    python scripts/extract_frames.py --around 3 --fps 10   # только ±3 c вокруг размеченных нарушений
    python scripts/extract_frames.py --only X-1 X-2

Результат: workdir/frames/<видео>/<видео>_<секунды>.jpg + workdir/frames/frames_index.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import VIDEOS_DIR, WORKDIR  # noqa: E402
from src.labels import load_labels, norm_name  # noqa: E402
from src.video_io import iter_frames, probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(VIDEOS_DIR))
    ap.add_argument("--out", default=str(WORKDIR / "frames"))
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--max-side", type=int, default=None, help="например 960 — уменьшить кадры")
    ap.add_argument("--around", type=float, default=None, help="только ±N секунд вокруг метки нарушения")
    ap.add_argument("--only", nargs="*", help="имена видео, напр. X-1 X-17")
    ap.add_argument("--quality", type=int, default=90)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in Path(a.videos).iterdir() if p.suffix.lower() in {".mov", ".mp4"})
    if a.only:
        keep = {norm_name(x) for x in a.only}
        videos = [v for v in videos if norm_name(v.name) in keep]
    labels = load_labels().set_index("key") if a.around else None

    rows = []
    for v in videos:
        key = norm_name(v.name)
        start, end = 0.0, None
        if labels is not None:
            tc = labels["timecode_sec"].get(key)
            if tc is None or pd.isna(tc):
                continue
            start, end = max(0.0, tc - a.around), tc + a.around
        meta = probe(v)
        d = out / key
        d.mkdir(exist_ok=True)
        n = 0
        for t, frame in iter_frames(v, a.fps, a.max_side, start, end):
            name = f"{key}_{t:07.2f}.jpg"
            cv2.imwrite(str(d / name), frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rows.append([key, v.name, f"{key}/{name}", t])
            n += 1
        print(f"{v.name}: {meta.fps:.2f} fps, {meta.duration_sec:.1f} c -> {n} кадров")

    with open(out / "frames_index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["key", "video", "frame", "timestamp_sec"], *rows])
    print(f"Итого {len(rows)} кадров -> {out}")


if __name__ == "__main__":
    main()
