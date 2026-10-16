"""NLP-baseline для текстов (обращения клиентов, жалобы, чаты): TF-IDF (слова + символы) + LogReg.

Символьные n-граммы отлично переживают смесь русского/казахского, опечатки и транслит.
Работает за секунды и часто даёт 80–90% качества трансформеров. Потом уже fine-tune BERT.

    python -m src.text_baseline --train data/raw/text_train.csv --test data/raw/text_test.csv \
        --text text --target label --id id
"""
from __future__ import annotations

import argparse
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union

from src.utils import ROOT, read_table


def clean(t: str) -> str:
    t = str(t).lower().replace("ё", "е")
    t = re.sub(r"http\S+|www\.\S+", " url ", t)
    t = re.sub(r"\+?\d[\d\s\-()]{8,}\d", " phone ", t)   # телефоны
    t = re.sub(r"\b\d{16}\b|\b\d{4}(\s\d{4}){3}\b", " cardnum ", t)  # номера карт (не тащим PII в модель)
    t = re.sub(r"\d+", " 0 ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_pipeline() -> object:
    feats = make_union(
        TfidfVectorizer(preprocessor=clean, analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(preprocessor=clean, analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                        sublinear_tf=True, max_features=200_000),
    )
    return make_pipeline(feats, LogisticRegression(max_iter=3000, C=5, class_weight="balanced"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test")
    ap.add_argument("--text", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--id")
    a = ap.parse_args()

    tr = read_table(a.train).dropna(subset=[a.text, a.target])
    pipe = build_pipeline()
    oof = cross_val_predict(pipe, tr[a.text].astype(str), tr[a.target],
                            cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)
    print(f"OOF F1-macro: {f1_score(tr[a.target], oof, average='macro'):.4f}")
    print(classification_report(tr[a.target], oof, digits=3))

    pipe.fit(tr[a.text].astype(str), tr[a.target])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "text_model.joblib")
    print("Модель: models/text_model.joblib")

    if a.test:
        te = read_table(a.test)
        sub = pd.DataFrame({a.id: te[a.id]} if a.id else {})
        sub[a.target] = pipe.predict(te[a.text].fillna("").astype(str))
        (ROOT / "submissions").mkdir(exist_ok=True)
        sub.to_csv(ROOT / "submissions" / "sub_text_baseline.csv", index=False)
        print("Сабмит: submissions/sub_text_baseline.csv")


if __name__ == "__main__":
    main()
