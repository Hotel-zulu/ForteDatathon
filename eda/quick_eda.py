"""EDA одной командой -> reports/eda_report.html (открывается в браузере, можно кидать в чат команды).

    python eda/quick_eda.py data/raw/train.csv --target target --test data/raw/test.csv
    python eda/quick_eda.py data/raw/train.csv --target target --profiling   # + ydata-profiling, если установлен

Что внутри: размеры и типы, пропуски, дубликаты, распределение таргета, числовые распределения,
топ категорий, корреляции, связь признаков с таргетом и ADVERSARIAL VALIDATION (похож ли test на train).
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT, read_table  # noqa: E402

plt.rcParams.update({"figure.dpi": 90, "axes.spines.top": False, "axes.spines.right": False})


def fig_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table_html(df: pd.DataFrame) -> str:
    return df.to_html(classes="t", border=0, float_format=lambda x: f"{x:,.4g}")


def overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_%": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
        "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    }).sort_values("missing_%", ascending=False)


def target_section(df, target):
    y = df[target]
    fig, ax = plt.subplots(figsize=(6, 3))
    if y.nunique() <= 30:
        vc = y.value_counts()
        vc.plot.bar(ax=ax, color="#3b6fd8")
        txt = table_html((vc / len(y) * 100).round(2).rename("share_%").to_frame())
        if len(vc) == 2 and vc.min() / vc.sum() < 0.1:
            txt += "<p class=warn>⚠ Сильный дисбаланс классов: используйте AUC/PR-AUC, stratified CV, подбор порога.</p>"
    else:
        y.plot.hist(ax=ax, bins=50, color="#3b6fd8")
        txt = table_html(y.describe().to_frame())
        if y.skew() > 2:
            txt += "<p class=warn>⚠ Таргет сильно скошен — попробуйте обучать на log1p(y).</p>"
    ax.set_title(f"Target: {target}")
    return fig_html(fig) + txt


def numeric_section(df, target):
    num = [c for c in df.select_dtypes("number").columns if c != target][:24]
    if not num:
        return "<p>Нет числовых признаков</p>"
    n = len(num)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(14, 2.6 * np.ceil(n / cols)))
    for ax, c in zip(np.ravel(axes), num):
        s = df[c].dropna()
        lo, hi = s.quantile([0.005, 0.995])
        s.clip(lo, hi).plot.hist(ax=ax, bins=40, color="#5a8f5a")
        ax.set_title(c, fontsize=9)
        ax.set_ylabel("")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig_html(fig) + table_html(df[num].describe().T)


def categorical_section(df, target):
    cats = [c for c in df.columns if df[c].dtype == object and c != target]
    out = []
    for c in cats[:15]:
        vc = df[c].value_counts(dropna=False).head(10)
        block = pd.DataFrame({"count": vc, "share_%": (vc / len(df) * 100).round(2)})
        if target in df and df[target].nunique() == 2 and pd.api.types.is_numeric_dtype(df[target]):
            block["target_rate"] = df.groupby(c, dropna=False)[target].mean().reindex(vc.index).round(4)
        out.append(f"<h3>{c} <small>({df[c].nunique()} уникальных)</small></h3>" + table_html(block))
    return "".join(out) or "<p>Нет категориальных признаков</p>"


def correlation_section(df, target):
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return ""
    corr = num.corr(method="spearman")
    k = min(len(corr), 25)
    corr = corr.iloc[:k, :k]
    fig, ax = plt.subplots(figsize=(0.45 * k + 3, 0.45 * k + 2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(k), corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    html = fig_html(fig)
    if target in num:
        top = num.corr(method="spearman")[target].drop(target).abs().sort_values(ascending=False).head(15)
        html += "<h3>|Spearman| с таргетом</h3>" + table_html(top.round(4).to_frame())
        if (top > 0.95).any():
            html += "<p class=warn>⚠ Есть признак почти идеально коррелирующий с таргетом — проверьте на УТЕЧКУ!</p>"
    return html


def adversarial_validation(train, test, target, id_col):
    """Учим модель отличать train от test. AUC≈0.5 — хорошо; AUC>0.7 — распределения разные."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    cols = [c for c in train.columns if c in test.columns and c not in {target, id_col}]
    both = pd.concat([train[cols], test[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    X = pd.DataFrame(index=both.index)
    for c in cols:
        X[c] = both[c] if pd.api.types.is_numeric_dtype(both[c]) else both[c].astype("category").cat.codes
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    p = cross_val_predict(m, X, y, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    m.fit(X, y)
    from sklearn.inspection import permutation_importance
    idx = X.sample(min(3000, len(X)), random_state=0).index
    pi = permutation_importance(m, X.loc[idx], y[idx], n_repeats=3, random_state=0)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False).head(10)
    verdict = ("✅ train и test похожи — обычный KFold подходит" if auc < 0.6 else
               "⚠ Заметный сдвиг. Посмотрите признаки ниже (часто это даты/id) — их стоит выбросить или валидироваться по времени")
    return f"<p><b>Adversarial AUC = {auc:.3f}</b>. {verdict}</p>" + table_html(imp.round(4).to_frame("drift_importance"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("--target")
    ap.add_argument("--test")
    ap.add_argument("--id", dest="id_col")
    ap.add_argument("--out", default=str(ROOT / "reports" / "eda_report.html"))
    ap.add_argument("--profiling", action="store_true", help="дополнительно ydata-profiling (медленно)")
    a = ap.parse_args()

    df = read_table(a.train)
    test = read_table(a.test) if a.test else None
    dups = int(df.duplicated().sum())
    sections = [
        ("Обзор", f"<p>Строк: <b>{len(df):,}</b>, колонок: <b>{df.shape[1]}</b>, "
                 f"дубликатов: <b>{dups}</b>, память: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
                 + (f"; test: <b>{test.shape}</b>" if test is not None else "") + "</p>"
                 + table_html(overview(df))),
        ("Пример данных", table_html(df.head(10))),
    ]
    if a.target:
        sections.append(("Таргет", target_section(df, a.target)))
    sections += [
        ("Числовые признаки", numeric_section(df, a.target)),
        ("Категориальные признаки", categorical_section(df, a.target)),
        ("Корреляции", correlation_section(df, a.target)),
    ]
    if test is not None:
        sections.append(("Adversarial validation (train vs test)", adversarial_validation(df, test, a.target, a.id_col)))

    css = """body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#1d2433}
    h2{border-bottom:2px solid #3b6fd8;padding-bottom:4px;margin-top:40px}
    .t{border-collapse:collapse;font-size:13px;margin:8px 0}.t td,.t th{padding:4px 10px;border-bottom:1px solid #e3e6ee;text-align:right}
    .t th{background:#f4f6fb}.warn{background:#fff4e0;padding:8px 12px;border-left:4px solid #f0a020}img{max-width:100%}
    nav a{margin-right:14px}"""
    nav = "".join(f'<a href="#s{i}">{t}</a>' for i, (t, _) in enumerate(sections))
    body = "".join(f'<h2 id="s{i}">{t}</h2>{h}' for i, (t, h) in enumerate(sections))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<html><head><meta charset=utf-8><title>EDA</title><style>{css}</style></head>"
                   f"<body><h1>EDA: {Path(a.train).name}</h1><nav>{nav}</nav>{body}</body></html>", encoding="utf-8")
    print(f"Отчёт: {out}")

    if a.profiling:
        try:
            from ydata_profiling import ProfileReport
            ProfileReport(df, minimal=len(df) > 100_000, title="Profiling").to_file(out.with_name("profiling.html"))
            print(f"Profiling: {out.with_name('profiling.html')}")
        except ImportError:
            print("ydata-profiling не установлен: pip install ydata-profiling")


if __name__ == "__main__":
    main()
