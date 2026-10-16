"""Подготовка признаков.

`add_features` — сюда пишете свой feature engineering под конкретную задачу.
`TablePrep`   — универсальная предобработка: типы, категории, даты. Обучается на train,
                применяется к test / к данным из API одинаково (никакого рассинхрона!).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Место для ручных фичей. Вызывается и при обучении, и в API/Streamlit.

    Идеи для банковских данных (раскомментируйте/адаптируйте):
    """
    df = df.copy()
    # --- Отношения (debt-to-income и т.п.) — часто самые сильные фичи в кредитном скоринге
    # df["dti"] = df["monthly_debt"] / (df["income"] + 1)
    # df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    # --- Агрегаты по клиенту из транзакций (если есть отдельная таблица транзакций):
    # agg = tx.groupby("client_id")["amount"].agg(["mean", "std", "max", "count"]).add_prefix("tx_")
    # df = df.merge(agg, on="client_id", how="left")
    # --- Частотное кодирование категорий
    # for c in ["region", "product"]:
    #     df[f"{c}_freq"] = df[c].map(df[c].value_counts(normalize=True))
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    """ISO (2024-01-31) парсим как есть, иначе считаем формат dd.mm.yyyy (принят в РК)."""
    s = s.astype("string")
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}").fillna(False).mean() > 0.5
    return pd.to_datetime(s, errors="coerce", dayfirst=not iso, format="mixed")


class TablePrep:
    """Fit/transform-предобработка, которую можно сохранить в joblib вместе с моделью."""

    def __init__(self, target: str, id_col: str | None = None, drop_cols: list[str] | None = None,
                 text_cols: list[str] | None = None, max_cat_unique: int = 1000):
        self.target = target
        self.id_col = id_col
        self.drop_cols = list(drop_cols or [])
        self.text_cols = list(text_cols or [])
        self.max_cat_unique = max_cat_unique
        self.features: list[str] = []
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.date_cols: list[str] = []
        self.categories: dict[str, list] = {}
        self.num_medians: dict[str, float] = {}

    # ---------- helpers ----------
    @staticmethod
    def _looks_like_date(s: pd.Series) -> bool:
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            return False
        sample = s.dropna().astype(str).head(200)
        if len(sample) == 0 or not sample.str.contains(r"\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}").mean() > 0.9:
            return False
        return _parse_dates(sample).notna().mean() > 0.9

    def _expand_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self.date_cols:
            d = _parse_dates(df[c])
            df[f"{c}_year"] = d.dt.year
            df[f"{c}_month"] = d.dt.month
            df[f"{c}_dow"] = d.dt.dayofweek
            df[f"{c}_day"] = d.dt.day
            df = df.drop(columns=c)
        return df

    # ---------- API ----------
    def fit(self, df: pd.DataFrame) -> "TablePrep":
        df = add_features(df)
        skip = {self.target, self.id_col, *self.drop_cols, *self.text_cols} - {None}
        cols = [c for c in df.columns if c not in skip]
        self.date_cols = [c for c in cols if self._looks_like_date(df[c])]
        df = self._expand_dates(df[cols])
        self.cat_cols, self.num_cols = [], []
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                self.num_cols.append(c)
            elif pd.api.types.is_numeric_dtype(s):
                self.num_cols.append(c)
            elif s.nunique() <= self.max_cat_unique:
                self.cat_cols.append(c)
            # иначе — слишком высокая кардинальность (свободный текст / уникальные id): выбрасываем
        for c in self.cat_cols:
            self.categories[c] = sorted(df[c].dropna().astype(str).unique().tolist())
        for c in self.num_cols:
            self.num_medians[c] = float(pd.to_numeric(df[c], errors="coerce").median())
        self.features = self.num_cols + self.cat_cols
        dropped = [c for c in df.columns if c not in self.features]
        if dropped:
            print(f"[TablePrep] Отброшены (высокая кардинальность): {dropped}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_features(df)
        for c in self.date_cols:
            if c not in df:
                df[c] = np.nan
        df = self._expand_dates(df)
        out = pd.DataFrame(index=df.index)
        for c in self.num_cols:
            out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
            out[c] = out[c].astype("float32")
        for c in self.cat_cols:
            vals = df[c].astype("string") if c in df else pd.Series(pd.NA, index=df.index, dtype="string")
            # неизвестные категории (новые в test/API) превращаются в NaN — модели это переживут
            out[c] = pd.Categorical(vals, categories=self.categories[c])
        return out[self.features]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
