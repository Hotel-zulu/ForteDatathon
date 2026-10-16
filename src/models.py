"""Фабрика моделей с единым интерфейсом fit / predict.

Поддерживаются: lgb, xgb, cb (catboost), hgb (sklearn HistGradientBoosting — работает всегда,
даже если бустинги не установились), linear (логрег / ridge — быстрый sanity-baseline).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def available(name: str) -> bool:
    mod = {"lgb": "lightgbm", "xgb": "xgboost", "cb": "catboost"}.get(name)
    return mod is None or importlib.util.find_spec(mod) is not None


def resolve_models(names: list[str]) -> list[str]:
    ok = [n for n in names if available(n)]
    missing = sorted(set(names) - set(ok))
    if missing:
        print(f"[models] Не установлены: {missing} — пропускаю")
    if not ok:
        print("[models] Ни одного бустинга нет, использую sklearn hgb")
        ok = ["hgb"]
    return ok


class Model:
    """Обёртка над конкретной библиотекой. task: binary | multiclass | regression."""

    def __init__(self, name: str, task: str, cat_cols: list[str], n_classes: int = 2,
                 seed: int = 42, params: dict | None = None):
        self.name, self.task, self.cat_cols = name, task, cat_cols
        self.n_classes, self.seed, self.params = n_classes, seed, params or {}
        self.model = None

    # ---- приведение категорий под каждую библиотеку ----
    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.name == "cb":
            X = X.copy()
            for c in self.cat_cols:
                X[c] = X[c].astype(str)  # NaN -> "nan", CatBoost так любит
            return X
        if self.name == "hgb":
            X = X.copy()
            for c in self.cat_cols:
                X[c] = X[c].cat.codes.replace(-1, np.nan).astype("float32")
            return X
        return X  # lgb / xgb / linear принимают pandas category

    def fit(self, X, y, X_val=None, y_val=None):
        X = self._X(X)
        Xv = self._X(X_val) if X_val is not None else None
        clf = self.task != "regression"
        p = dict(self.params)

        if self.name == "lgb":
            import lightgbm as lgb
            base = dict(n_estimators=5000, learning_rate=0.03, num_leaves=63, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, random_state=self.seed, verbose=-1)
            base.update(p)
            self.model = (lgb.LGBMClassifier if clf else lgb.LGBMRegressor)(**base)
            kw = {}
            if Xv is not None:
                kw = dict(eval_set=[(Xv, y_val)], callbacks=[lgb.early_stopping(200, verbose=False)])
            self.model.fit(X, y, **kw)

        elif self.name == "xgb":
            import xgboost as xgb
            base = dict(n_estimators=5000, learning_rate=0.03, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                        random_state=self.seed, early_stopping_rounds=200 if Xv is not None else None)
            base.update(p)
            self.model = (xgb.XGBClassifier if clf else xgb.XGBRegressor)(**base)
            self.model.fit(X, y, eval_set=[(Xv, y_val)] if Xv is not None else None, verbose=False)

        elif self.name == "cb":
            import catboost as cb
            base = dict(iterations=5000, learning_rate=0.05, depth=6, random_seed=self.seed,
                        verbose=0, cat_features=self.cat_cols, allow_writing_files=False)
            if self.task == "multiclass":
                base["loss_function"] = "MultiClass"
            base.update(p)
            self.model = (cb.CatBoostClassifier if clf else cb.CatBoostRegressor)(**base)
            kw = dict(eval_set=(Xv, y_val), early_stopping_rounds=200) if Xv is not None else {}
            self.model.fit(X, y, **kw)

        elif self.name == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            base = dict(max_iter=1000, learning_rate=0.05, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=50, random_state=self.seed)
            base.update(p)
            self.model = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(**base)
            self.model.fit(X, y)

        elif self.name == "linear":
            from sklearn.linear_model import LogisticRegression, Ridge
            num = [c for c in X.columns if c not in self.cat_cols]
            pre = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                      OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
                 self.cat_cols),
            ])
            est = LogisticRegression(max_iter=2000, C=1.0) if clf else Ridge(alpha=1.0)
            self.model = make_pipeline(pre, est)
            Xl = X.copy()
            for c in self.cat_cols:
                Xl[c] = Xl[c].astype(object).where(Xl[c].notna(), "NA")
            self.model.fit(Xl, y)
        else:
            raise ValueError(f"Неизвестная модель {self.name}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """binary -> P(class=1); multiclass -> матрица вероятностей; regression -> значения."""
        X = self._X(X)
        if self.name == "linear":
            X = X.copy()
            for c in self.cat_cols:
                X[c] = X[c].astype(object).where(X[c].notna(), "NA")
        if self.task == "regression":
            return np.asarray(self.model.predict(X), dtype=float)
        proba = np.asarray(self.model.predict_proba(X))
        return proba[:, 1] if self.task == "binary" else proba

    def feature_importance(self, columns: list[str]) -> pd.Series | None:
        m = self.model
        if self.name == "cb":
            return pd.Series(m.get_feature_importance(), index=columns)
        if hasattr(m, "feature_importances_"):
            return pd.Series(m.feature_importances_, index=columns)
        return None
