"""Метрики. Всегда уточняйте у организаторов, какая метрика в лидерборде, и считайте ровно её."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as M

# (функция, чем больше — тем лучше?, нужны ли вероятности)
METRICS = {
    "roc_auc": (lambda y, p: M.roc_auc_score(y, p, multi_class="ovr") if np.ndim(p) == 2 else M.roc_auc_score(y, p), True),
    "gini": (lambda y, p: 2 * M.roc_auc_score(y, p) - 1, True),  # банки любят Gini = 2*AUC-1
    "pr_auc": (lambda y, p: M.average_precision_score(y, p), True),
    "logloss": (lambda y, p: M.log_loss(y, p), False),
    "f1": (lambda y, p: M.f1_score(y, _labels(p)), True),
    "f1_macro": (lambda y, p: M.f1_score(y, _labels(p), average="macro"), True),
    "f1_weighted": (lambda y, p: M.f1_score(y, _labels(p), average="weighted"), True),
    "accuracy": (lambda y, p: M.accuracy_score(y, _labels(p)), True),
    "rmse": (lambda y, p: float(np.sqrt(M.mean_squared_error(y, p))), False),
    "mae": (lambda y, p: M.mean_absolute_error(y, p), False),
    "mape": (lambda y, p: M.mean_absolute_percentage_error(y, p), False),
    "r2": (lambda y, p: M.r2_score(y, p), True),
}


def _labels(p, thr: float = 0.5):
    p = np.asarray(p)
    return p.argmax(1) if p.ndim == 2 else (p >= thr).astype(int)


def default_metric(task: str) -> str:
    return {"binary": "roc_auc", "multiclass": "f1_macro", "regression": "rmse"}[task]


def score(name: str, y, p) -> float:
    return float(METRICS[name][0](y, p))


def greater_is_better(name: str) -> bool:
    return METRICS[name][1]


def best_threshold(y, p, metric=M.f1_score) -> tuple[float, float]:
    """Подбор порога для бинарной задачи по OOF (нужен, если метрика — F1/accuracy, а не AUC)."""
    best = (0.5, -1.0)
    for t in np.linspace(0.05, 0.95, 91):
        s = metric(y, (p >= t).astype(int))
        if s > best[1]:
            best = (float(t), float(s))
    return best
