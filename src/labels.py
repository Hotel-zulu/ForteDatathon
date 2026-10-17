"""Разметка организаторов: open_labels.xlsx -> configs/labels.csv (имя файла, секунда, метка).

Нормализует:
  * таймкоды вида «22 секунда», «1минута 8 секунда» -> 22.0, 68.0
  * имена файлов: в датасете смешаны латинская X (X-1, X-2) и кириллическая Х (Х-3...) — приводим к латинице
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.config import LABELS_CSV, LABELS_XLSX, VIDEOS_DIR

CYR2LAT = str.maketrans({"Х": "X", "х": "x", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
                         "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T"})


def norm_name(name: str) -> str:
    name = unicodedata.normalize("NFC", str(name)).strip()
    return Path(name).stem.translate(CYR2LAT).upper()


def parse_timecode(s) -> float | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).lower().replace(",", ".")
    if re.fullmatch(r"\s*\d+(\.\d+)?\s*", s):
        return float(s)
    if m := re.fullmatch(r"\s*(\d+):(\d{1,2}(?:\.\d+)?)\s*", s):
        return int(m[1]) * 60 + float(m[2])
    minutes = re.search(r"(\d+)\s*мин", s)
    seconds = re.search(r"(\d+(?:\.\d+)?)\s*сек", s)
    if not minutes and not seconds:
        return None
    return (int(minutes[1]) * 60 if minutes else 0) + (float(seconds[1]) if seconds else 0)


def load_labels(xlsx: Path = LABELS_XLSX, videos_dir: Path = VIDEOS_DIR) -> pd.DataFrame:
    df = pd.read_excel(xlsx)
    df.columns = ["name", "timecode_raw", "label_raw"][: len(df.columns)]
    df["key"] = df["name"].map(norm_name)
    df["timecode_sec"] = df["timecode_raw"].map(parse_timecode)
    df["violation"] = df["label_raw"].astype(str).str.strip().str.lower().eq("нарушение").astype(int)
    files = {norm_name(p.name): p.name for p in Path(videos_dir).glob("*") if p.suffix.lower() in {".mov", ".mp4"}}
    df["file"] = df["key"].map(files)
    return df[["key", "file", "violation", "timecode_sec", "name", "timecode_raw", "label_raw"]]


if __name__ == "__main__":
    d = load_labels()
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(LABELS_CSV, index=False)
    print(d.to_string(index=False))
    print(f"\nВсего {len(d)}: нарушений {d.violation.sum()}, без нарушений {(1 - d.violation).sum()}")
    if d["file"].isna().any():
        print("⚠ Нет видеофайла для:", d.loc[d.file.isna(), "name"].tolist())
    bad = d[(d.violation == 1) & d.timecode_sec.isna()]
    if len(bad):
        print("⚠ Не распарсился таймкод:", bad[["name", "timecode_raw"]].values.tolist())
    print(f"Сохранено: {LABELS_CSV}")
