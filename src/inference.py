"""Единая функция предсказания — её используют и Streamlit, и FastAPI, и predict-скрипт."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils import ROOT

DEFAULT_BUNDLE = ROOT / "models" / "bundle.joblib"


@lru_cache(maxsize=2)
def load_bundle(path: str | Path = DEFAULT_BUNDLE) -> dict:
    return joblib.load(path)


def predict_df(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Возвращает исходный df + колонки prediction (и probability / proba_<class>)."""
    b = bundle or load_bundle()
    X = b["prep"].transform(df)
    preds = [m.predict(X) for fold_models in b["models"].values() for m in fold_models]
    p = np.mean(preds, axis=0)

    out = df.copy()
    task, classes = b["task"], b["classes"]
    if task == "regression":
        out["prediction"] = p
    elif task == "binary":
        thr = b.get("threshold") or 0.5
        out["probability"] = p.round(4)
        out["prediction"] = np.where(p >= thr, classes[1], classes[0])
    else:
        for i, c in enumerate(classes):
            out[f"proba_{c}"] = p[:, i].round(4)
        out["prediction"] = [classes[i] for i in p.argmax(1)]
    return out


def model_info(bundle: dict | None = None) -> dict:
    b = bundle or load_bundle()
    prep = b["prep"]
    return dict(task=b["task"], metric=b["metric"], scores=b["scores"], classes=b["classes"],
                threshold=b.get("threshold"), models=list(b["models"].keys()),
                num_features={c: prep.num_medians.get(c) for c in prep.num_cols},
                cat_features={c: prep.categories[c][:50] for c in prep.cat_cols},
                date_features=prep.date_cols)


if __name__ == "__main__":
    import argparse
    from src.utils import read_table

    ap = argparse.ArgumentParser(description="Предсказание по файлу с сохранённым бандлом")
    ap.add_argument("input")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    predict_df(read_table(a.input)).to_csv(a.out, index=False)
    print(f"Сохранено: {a.out}")
