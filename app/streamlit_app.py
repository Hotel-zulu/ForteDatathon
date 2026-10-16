"""Демо-интерфейс для защиты.  Запуск:  streamlit run app/streamlit_app.py

Вкладки: пакетный скоринг файла / ручной ввод одного клиента / о модели.
Работает с любым бандлом из src.train — форма ввода строится автоматически по признакам модели.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import DEFAULT_BUNDLE, load_bundle, model_info, predict_df  # noqa: E402
from src.metrics import greater_is_better  # noqa: E402
from src.utils import read_table  # noqa: E402

st.set_page_config(page_title="Forte AI — демо", page_icon="🏦", layout="wide")

# ---------- Шапка (поменяйте под свой кейс) ----------
st.title("🏦 Название решения")
st.caption("Одной строкой: какую бизнес-проблему банка решаем и для кого (кредитный аналитик / оператор колл-центра / клиент)")

if not DEFAULT_BUNDLE.exists():
    st.error("Модель не найдена. Сначала обучите: `python -m src.train --train ... --target ...`")
    st.stop()

bundle = load_bundle()
info = model_info(bundle)

with st.sidebar:
    st.header("Модель")
    best = (max if greater_is_better(info["metric"]) else min)(info["scores"].values())
    st.metric(f"OOF {info['metric']}", f"{best:.4f}")
    st.write("Модели:", ", ".join(info["models"]))
    if info["task"] == "binary":
        thr = st.slider("Порог решения", 0.0, 1.0, float(info["threshold"] or 0.5), 0.01,
                        help="Ниже порог — ловим больше дефолтов, но больше ложных отказов")
        bundle = {**bundle, "threshold": thr}

tab_file, tab_one, tab_about = st.tabs(["📄 Скоринг файла", "👤 Один клиент", "ℹ️ О модели"])

# ---------- 1. Пакетный скоринг ----------
with tab_file:
    up = st.file_uploader("Загрузите CSV / XLSX / Parquet", type=["csv", "xlsx", "parquet"])
    if up:
        tmp = Path("/tmp") / up.name
        tmp.write_bytes(up.getvalue())
        df = read_table(tmp)
        with st.spinner("Считаем..."):
            res = predict_df(df, bundle)
        c1, c2, c3 = st.columns(3)
        c1.metric("Записей", f"{len(res):,}")
        if "probability" in res:
            c2.metric("Средняя вероятность", f"{res['probability'].mean():.1%}")
            c3.metric("Отмечено моделью", f"{(res['prediction'] == info['classes'][1]).mean():.1%}")
            st.bar_chart(pd.cut(res["probability"], bins=10).value_counts().sort_index().rename(lambda x: f"{x.left:.1f}-{x.right:.1f}"))
        elif info["task"] != "regression":
            st.bar_chart(res["prediction"].value_counts())
        st.dataframe(res, use_container_width=True, height=400)
        st.download_button("⬇️ Скачать результат", res.to_csv(index=False).encode("utf-8-sig"),
                           "predictions.csv", "text/csv")

# ---------- 2. Ручной ввод ----------
with tab_one:
    prep = bundle["prep"]
    date_derived = {f"{d}_{s}" for d in prep.date_cols for s in ("year", "month", "dow", "day")}
    row = {}
    cols = st.columns(3)
    i = 0
    for c in prep.num_cols:
        if c in date_derived:
            continue
        default = prep.num_medians.get(c) or 0.0
        row[c] = cols[i % 3].number_input(c, value=float(default) if pd.notna(default) else 0.0)
        i += 1
    for c in prep.cat_cols:
        row[c] = cols[i % 3].selectbox(c, prep.categories[c])
        i += 1
    for c in prep.date_cols:
        row[c] = str(cols[i % 3].date_input(c))
        i += 1
    if st.button("Предсказать", type="primary"):
        res = predict_df(pd.DataFrame([row]), bundle).iloc[0]
        if "probability" in res:
            st.metric("Вероятность", f"{res['probability']:.1%}")
            (st.error if res["prediction"] == info["classes"][1] else st.success)(f"Решение модели: {res['prediction']}")
        else:
            st.success(f"Предсказание: {res['prediction']}")
        # TODO: сюда добавьте объяснение (SHAP waterfall) — жюри банка очень любит «почему модель так решила»

# ---------- 3. О модели ----------
with tab_about:
    st.json({k: v for k, v in info.items() if k not in ("num_features", "cat_features")})
    fi = Path(__file__).resolve().parents[1] / "reports" / "feature_importance.csv"
    if fi.exists():
        st.subheader("Важность признаков")
        st.bar_chart(pd.read_csv(fi, index_col=0).head(20))
