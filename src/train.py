"""Baseline за одну команду: CV + несколько моделей + бленд + сабмит + сохранённый бандл для API.

Примеры:
    python -m src.train --train data/raw/train.csv --test data/raw/test.csv --target target --id id
    python -m src.train ... --models lgb,cb --folds 5 --metric gini
    python -m src.train ... --group client_id          # GroupKFold, если у клиента несколько строк
    python -m src.train ... --time-col report_date     # TimeSeriesSplit, если данные во времени
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from src.features import TablePrep
from src.metrics import best_threshold, default_metric, greater_is_better, score
from src.models import Model, resolve_models
from src.utils import ROOT, read_table, seed_everything, timer


def detect_task(y: pd.Series) -> str:
    if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
        return "regression"
    n = y.nunique()
    if n == 2:
        return "binary"
    if n <= 50 or not pd.api.types.is_numeric_dtype(y):
        return "multiclass"
    return "regression"


def make_splits(X, y, task, folds, seed, groups=None, time_order=None):
    if time_order is not None:
        order = np.argsort(time_order.values, kind="stable")
        for tr, va in TimeSeriesSplit(n_splits=folds).split(order):
            yield order[tr], order[va]
        return
    if groups is not None:
        yield from GroupKFold(n_splits=folds).split(X, y, groups)
        return
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed) if task != "regression" \
        else KFold(folds, shuffle=True, random_state=seed)
    yield from cv.split(X, y)


def run(train_path, target, test_path=None, id_col=None, models="lgb,cb,xgb", folds=5,
        metric=None, task=None, group=None, time_col=None, drop=None, seed=42, tag="baseline"):
    seed_everything(seed)
    train = read_table(train_path)
    test = read_table(test_path) if test_path else None
    print(f"train {train.shape}" + (f", test {test.shape}" if test is not None else ""))

    train = train[train[target].notna()].reset_index(drop=True)
    y_raw = train[target]
    task = task or detect_task(y_raw)
    metric = metric or default_metric(task)
    classes = None
    if task == "regression":
        y = y_raw.astype(float).values
    else:
        classes = sorted(y_raw.unique().tolist(), key=str)
        y = y_raw.map({c: i for i, c in enumerate(classes)}).astype(int).values
    print(f"Задача: {task}, метрика: {metric}" + (f", классы: {classes}" if classes else ""))

    drop_cols = list(drop or [])
    if time_col:
        drop_cols.append(time_col)  # время — для сплита, а не как признак (иначе утечка)
    prep = TablePrep(target=target, id_col=id_col, drop_cols=drop_cols + ([group] if group else []))
    X = prep.fit_transform(train)
    X_test = prep.transform(test) if test is not None else None
    print(f"Признаков: {len(prep.features)} (num {len(prep.num_cols)}, cat {len(prep.cat_cols)})")

    n_cls = len(classes) if classes else 1
    shape = (len(X), n_cls) if task == "multiclass" else (len(X),)
    names = resolve_models([m.strip() for m in models.split(",") if m.strip()])
    oof, test_pred, fitted, results, importances = {}, {}, {}, {}, []

    groups = train[group] if group else None
    time_order = pd.to_datetime(train[time_col], errors="coerce") if time_col else None

    for name in names:
        oof[name] = np.full(shape, np.nan)
        test_pred[name] = 0 if X_test is not None else None
        fitted[name] = []
        with timer(name):
            splits = list(make_splits(X, y, task, folds, seed, groups, time_order))
            for k, (tr, va) in enumerate(splits):
                m = Model(name, task, prep.cat_cols, n_classes=n_cls, seed=seed + k)
                m.fit(X.iloc[tr], y[tr], X.iloc[va], y[va])
                oof[name][va] = m.predict(X.iloc[va])
                if X_test is not None:
                    test_pred[name] = test_pred[name] + m.predict(X_test) / len(splits)
                fitted[name].append(m)
                imp = m.feature_importance(prep.features)
                if imp is not None:
                    importances.append(imp.rename(f"{name}_{k}"))
                print(f"  fold {k}: {score(metric, y[va], oof[name][va]):.5f}")
        mask = ~np.isnan(oof[name] if oof[name].ndim == 1 else oof[name][:, 0])
        results[name] = score(metric, y[mask], oof[name][mask])
        print(f"==> {name} OOF {metric}: {results[name]:.5f}")

    # ---- Бленд: простое среднее моделей (на хакатоне почти всегда даёт +) ----
    mask = ~np.isnan(np.mean([oof[n] if oof[n].ndim == 1 else oof[n][:, 0] for n in names], axis=0))
    blend_oof = np.mean([oof[n] for n in names], axis=0)
    if len(names) > 1:
        results["blend"] = score(metric, y[mask], blend_oof[mask])
        print(f"==> BLEND OOF {metric}: {results['blend']:.5f}")

    threshold = None
    if task == "binary":
        threshold, f1 = best_threshold(y[mask], blend_oof[mask])
        print(f"Лучший порог по F1 на OOF: {threshold:.2f} (F1={f1:.4f})")

    # ---- Сохранение артефактов ----
    stamp = datetime.now().strftime("%m%d_%H%M")
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "submissions").mkdir(exist_ok=True)
    bundle = dict(prep=prep, task=task, classes=classes, threshold=threshold, metric=metric,
                  models=fitted, target=target, id_col=id_col, scores=results)
    joblib.dump(bundle, ROOT / "models" / "bundle.joblib")
    print(f"Бандл модели: models/bundle.joblib")

    if importances:
        fi = pd.concat(importances, axis=1)
        fi = (fi / fi.sum()).mean(axis=1).sort_values(ascending=False)
        fi.to_csv(ROOT / "reports" / "feature_importance.csv", header=["importance"])
        print("Топ-10 признаков:\n" + fi.head(10).round(4).to_string())

    sub_path = None
    if X_test is not None:
        pred = np.mean([test_pred[n] for n in names], axis=0)
        sub = pd.DataFrame({id_col: test[id_col]}) if id_col and id_col in test else pd.DataFrame(index=test.index)
        label_metrics = {"f1", "f1_macro", "f1_weighted", "accuracy"}
        if task == "regression":
            sub[target] = pred
        elif task == "binary":
            sub[target] = np.where(pred >= threshold, classes[1], classes[0]) if metric in label_metrics else pred
        else:
            if metric in label_metrics:
                sub[target] = [classes[i] for i in pred.argmax(1)]
            else:
                for i, c in enumerate(classes):
                    sub[f"{target}_{c}"] = pred[:, i]
        sub_path = ROOT / "submissions" / f"sub_{tag}_{stamp}_{metric}_{results.get('blend', results[names[-1]]):.4f}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"Сабмит: {sub_path.relative_to(ROOT)}")

    # ---- Журнал экспериментов: чтобы через 20 часов помнить, что сработало ----
    log = ROOT / "reports" / "experiments.csv"
    row = pd.DataFrame([dict(time=stamp, tag=tag, task=task, metric=metric, models=",".join(names),
                             folds=folds, n_features=len(prep.features),
                             scores=json.dumps({k: round(v, 5) for k, v in results.items()}),
                             submission=sub_path.name if sub_path else "")])
    row.to_csv(log, mode="a", header=not log.exists(), index=False)
    return bundle, oof, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test")
    ap.add_argument("--target", required=True)
    ap.add_argument("--id", dest="id_col")
    ap.add_argument("--models", default="lgb,cb,xgb", help="lgb,xgb,cb,hgb,linear")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--metric")
    ap.add_argument("--task", choices=["binary", "multiclass", "regression"])
    ap.add_argument("--group")
    ap.add_argument("--time-col")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    run(a.train, a.target, a.test, a.id_col, a.models, a.folds, a.metric, a.task, a.group,
        a.time_col, a.drop, a.seed, a.tag)


if __name__ == "__main__":
    main()
