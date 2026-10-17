"""Обзорный отчёт для команды: «что происходит в роликах».

    python scripts/review_report.py              # все 42 видео
    python scripts/review_report.py --no-yolo    # без детекции (быстрее)
    python scripts/review_report.py --only X-1 X-2

Результат:
  workdir/review/index.html   — таблица + раскадровки (открыть в браузере)
  workdir/review/summary.csv  — та же таблица (для Excel/Google Sheets, колонка «что видно» пустая — заполнить)
  workdir/review/frames/...   — JPEG-кадры

Для роликов с нарушением: 9 кадров вокруг метки (t-4 ... t+4 c) + обзор всего ролика.
Для роликов без нарушения: 9 кадров равномерно по ролику.
На кадрах рамки YOLO: люди (зелёные), телефон (красный), сумки (оранжевые) и т.п.
"""
from __future__ import annotations

import argparse
import html
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import VIDEOS_DIR, WORKDIR  # noqa: E402
from src.labels import load_labels, norm_name  # noqa: E402
from src.video_io import frame_at, probe, resize_max_side  # noqa: E402

COLORS = {"person": (60, 200, 60), "phone": (40, 40, 255), "handbag": (0, 160, 255), "backpack": (0, 160, 255),
          "suitcase": (0, 160, 255), "laptop": (255, 160, 0), "book": (200, 0, 200)}


def draw(img, obs):
    h, w = img.shape[:2]
    items = [("person", p) for p in obs.persons] + [(o["cls"], o) for o in obs.objects]
    for name, it in items:
        x1, y1, x2, y2 = (np.array(it["xyxy"]) * [w, h, w, h]).astype(int)
        col = COLORS.get(name, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"{name}{'#' + str(it['track_id']) if it.get('track_id') is not None else ''} {it['conf']:.2f}"
        cv2.putText(img, label, (x1 + 2, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--window", type=float, default=4.0, help="± секунд вокруг метки")
    ap.add_argument("--n", type=int, default=9, help="кадров в раскадровке")
    a = ap.parse_args()

    out = WORKDIR / "review"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    if a.only:
        keep = {norm_name(x) for x in a.only}
        labels = labels[labels.key.isin(keep)]
    labels = labels.assign(_n=labels.key.str.extract(r"(\d+)")[0].astype(int)).sort_values("_n")

    perception = None
    if not a.no_yolo:
        from src.detectors.yolo_perception import YoloPerception
        perception = YoloPerception(track=False)
        print(f"YOLO на устройстве: {perception.device}")

    rows, blocks = [], []
    for r in labels.itertuples():
        if not isinstance(r.file, str):
            continue
        path = VIDEOS_DIR / r.file
        m = probe(path)
        if r.violation and pd.notna(r.timecode_sec):
            ts = np.linspace(max(0, r.timecode_sec - a.window), min(m.duration_sec - 0.05, r.timecode_sec + a.window), a.n)
        else:
            ts = np.linspace(0.5, max(0.5, m.duration_sec - 0.5), a.n)
        overview = np.linspace(0.5, max(0.5, m.duration_sec - 0.5), 6) if r.violation else []

        cells, counts = [], Counter()
        for kind, series in (("main", ts), ("overview", overview)):
            for t in series:
                img = frame_at(path, float(t), max_side=960)
                if img is None:
                    continue
                if perception:
                    obs = perception(float(t), img)
                    if kind == "main":
                        counts["person_max"] = max(counts["person_max"], len(obs.persons))
                        for o in obs.objects:
                            counts[o["cls"]] += 1
                    img = draw(img, obs)
                img = resize_max_side(img, 480)
                is_mark = r.violation and abs(t - (r.timecode_sec or -99)) < 0.6 and kind == "main"
                fn = f"{r.key}_{kind}_{t:06.1f}.jpg"
                cv2.imwrite(str(out / "frames" / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                cells.append((kind, t, fn, is_mark))

        objs = ", ".join(f"{k}×{v}" for k, v in counts.items() if k != "person_max") or "—"
        rows.append(dict(video=r.key, file=r.file, label="нарушение" if r.violation else "без нарушения",
                         timecode_sec=r.timecode_sec, duration_sec=round(m.duration_sec, 1), fps=round(m.fps, 2),
                         resolution=f"{m.width}x{m.height}", max_persons=counts.get("person_max", ""),
                         objects_near_mark=objs, what_is_visible="", violation_type="", npa_clause=""))
        print(f"{r.key:5s} {rows[-1]['label']:14s} t={r.timecode_sec}  {m.duration_sec:.0f}s  {objs}")

        def strip(kind):
            return "".join(
                f'<figure class="{"mark" if mk else ""}"><a href="frames/{fn}" target=_blank><img loading=lazy src="frames/{fn}"></a>'
                f"<figcaption>{t:.1f} c{' ◀ метка' if mk else ''}</figcaption></figure>"
                for k, t, fn, mk in cells if k == kind)

        badge = f'<span class="b bad">нарушение · {r.timecode_sec:g} c</span>' if r.violation else '<span class="b ok">без нарушения</span>'
        blocks.append(f'<section id="{r.key}"><h2>{r.key} {badge} <small>{m.duration_sec:.0f} c · {m.width}×{m.height} · {m.fps:.1f} fps · {html.escape(objs)}</small></h2>'
                      f'<div class="strip">{strip("main")}</div>'
                      + (f'<details><summary>Обзор всего ролика</summary><div class="strip">{strip("overview")}</div></details>' if r.violation else "")
                      + "</section>")

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    table = df.drop(columns=["file", "what_is_visible", "violation_type", "npa_clause"]).to_html(index=False, border=0, classes="t", na_rep="—")
    for k in df.video:
        table = table.replace(f"<td>{k}</td>", f'<td><a href="#{k}">{k}</a></td>', 1)
    css = """body{font:14px system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1c2230}main{max-width:1500px;margin:auto;padding:20px 16px}
    .t{border-collapse:collapse;background:#fff}.t td,.t th{padding:5px 10px;border-bottom:1px solid #e4e7ec;text-align:left}.t th{background:#eef0f4}
    section{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:12px 16px;margin:16px 0}h2{font-size:17px;margin:0 0 10px}h2 small{color:#667085;font-weight:400}
    .strip{display:flex;gap:6px;overflow-x:auto}figure{margin:0;flex:0 0 auto}figure img{width:240px;border-radius:6px;display:block}
    figure.mark img{outline:4px solid #d92d20}figcaption{font-size:12px;color:#667085;text-align:center}.mark figcaption{color:#d92d20;font-weight:700}
    .b{font-size:12px;padding:2px 8px;border-radius:10px;margin-left:6px}.bad{background:#fdecee;color:#b4122b}.ok{background:#e7f6ec;color:#127a3d}"""
    (out / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>Обзор роликов</title><style>{css}</style><main>"
        f"<h1>Обзор роликов датасета</h1><p>Зелёные рамки — люди, красные — телефон, оранжевые — сумки (YOLO COCO). "
        f"Красная обводка — кадр метки нарушения. Клик по кадру открывает его крупно.</p>{table}{''.join(blocks)}</main>",
        encoding="utf-8")
    print(f"\nОтчёт: {out / 'index.html'}\nТаблица: {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
