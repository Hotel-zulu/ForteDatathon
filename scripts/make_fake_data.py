"""Генерирует синтетический «банковский» датасет, чтобы прогнать весь пайплайн ДО хакатона.

    python scripts/make_fake_data.py              # кредитный скоринг (binary)
    python scripts/make_fake_data.py --task text  # классификация обращений клиентов (NLP)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def credit(n=6000, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(n),
        "age": r.integers(18, 70, n),
        "income": r.lognormal(12.5, 0.6, n).round(0),           # тенге/мес
        "loan_amount": r.lognormal(14, 0.8, n).round(0),
        "term_months": r.choice([6, 12, 24, 36, 60], n),
        "region": r.choice(["Астана", "Алматы", "Шымкент", "Караганда", "Актобе", "Атырау"], n),
        "product": r.choice(["потреб", "авто", "ипотека", "карта"], n, p=[.5, .15, .1, .25]),
        "employment": r.choice(["наём", "ИП", "госслужба", "безработный"], n, p=[.6, .2, .15, .05]),
        "n_prev_loans": r.poisson(1.5, n),
        "max_dpd_12m": r.choice([0, 0, 0, 5, 15, 30, 60, 90], n),
        "application_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(r.integers(0, 600, n), "D"),
    })
    df.loc[r.random(n) < 0.08, "income"] = np.nan
    logit = (-2.5 + 0.9 * np.log1p(df.loan_amount / df.income.fillna(df.income.median()))
             + 0.025 * df.max_dpd_12m - 0.015 * (df.age - 40)
             + (df.employment == "безработный") * 1.2 + (df["product"] == "карта") * 0.4)
    df["target"] = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["application_date"] = df["application_date"].dt.strftime("%Y-%m-%d")
    train, test = df.iloc[: int(n * .8)], df.iloc[int(n * .8):].drop(columns="target")
    return train, test


def text(n=3000, seed=0):
    r = np.random.default_rng(seed)
    topics = {
        "карта": ["заблокировали карту", "не работает карта в банкомате", "перевыпуск карты", "карта не пришла"],
        "перевод": ["перевод не дошел", "деньги списались но не пришли", "ошибка перевода по номеру", "kaspi перевод завис"],
        "кредит": ["хочу досрочно погасить кредит", "неверный график платежей", "рефинансирование кредита", "отказали в кредите"],
        "приложение": ["не могу войти в приложение", "приложение вылетает", "не приходит смс код", "ошибка в мобильном банке"],
    }
    tails = ["", " помогите пожалуйста", " срочно!!!", " уже третий день", " Сәлеметсіз бе", " очень жду ответа"]
    rows = []
    for i in range(n):
        label = r.choice(list(topics))
        rows.append({"id": i, "text": r.choice(topics[label]) + r.choice(tails), "label": label})
    df = pd.DataFrame(rows)
    return df.iloc[: int(n * .8)], df.iloc[int(n * .8):].drop(columns="label")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["credit", "text"], default="credit")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = credit() if a.task == "credit" else text()
    prefix = "" if a.task == "credit" else "text_"
    tr.to_csv(OUT / f"{prefix}train.csv", index=False)
    te.to_csv(OUT / f"{prefix}test.csv", index=False)
    print(f"Сохранено в {OUT}: {prefix}train.csv {tr.shape}, {prefix}test.csv {te.shape}")
