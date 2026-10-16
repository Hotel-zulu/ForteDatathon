"""Мелкие утилиты: фиксация seed, таймер, загрузка любых табличных файлов, экономия памяти."""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"[{name}] старт")
    yield
    print(f"[{name}] готово за {time.time() - t0:.1f} c")


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Читает csv / tsv / parquet / xlsx / json по расширению."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".csv", ".txt"}:
        # sep=None + engine=python сам угадает ; или , (частая боль с казахстанскими выгрузками)
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", **kwargs)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if ext in {".json", ".jsonl"}:
        return pd.read_json(path, lines=ext == ".jsonl", **kwargs)
    raise ValueError(f"Неизвестный формат файла: {path}")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Понижает типы чисел (float64->float32, int64->int8/16/32). Спасает на больших датасетах."""
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.select_dtypes(include=["int", "float"]).columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        else:
            df[col] = pd.to_numeric(s, downcast="float")
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Память: {start:.1f} MB -> {end:.1f} MB")
    return df
