# Forte Hackathon Starter 🏦

Готовый каркас для ML-хакатона: окружение, EDA одной командой, baseline с кросс-валидацией и блендом бустингов,
NLP-baseline, Streamlit-демо, FastAPI и Docker. Всё уже проверено на синтетических банковских данных.

## Быстрый старт (сделать ДО хакатона)

```bash
# 1. Окружение (Python 3.11!)
bash setup.sh                       # macOS/Linux
powershell -ExecutionPolicy Bypass -File setup.ps1   # Windows
# или conda:  conda env create -f environment.yml && conda activate forte

# 2. Проверка, что у всех всё стоит
python scripts/check_env.py

# 3. Прогон всего пайплайна на синтетике — убедиться, что ничего не падает
python scripts/make_fake_data.py
python eda/quick_eda.py data/raw/train.csv --target target --test data/raw/test.csv --id id
python -m src.train --train data/raw/train.csv --test data/raw/test.csv --target target --id id
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload       # Swagger: http://localhost:8000/docs
docker compose up --build           # API :8000 + демо :8501
```

## День хакатона: первые 30 минут

| Минута | Действие | Команда |
|---|---|---|
| 0–5 | Положить данные в `data/raw/`, прочитать описание метрики | — |
| 5–10 | EDA-отчёт → кинуть в чат команды | `python eda/quick_eda.py data/raw/train.csv --target <T> --test data/raw/test.csv --id <ID>` |
| 10–25 | Baseline + первый сабмит | `python -m src.train --train ... --test ... --target <T> --id <ID> --metric <как в лидерборде>` |
| 25–30 | Сабмит на платформу, сверить CV и лидерборд | — |

Если CV и лидерборд сильно расходятся — проблема в валидации, а не в модели (см. `--group` и `--time-col`).

## Структура

```
src/
  train.py          CV (KFold / Stratified / Group / TimeSeries) + lgb/cb/xgb/hgb/linear + бленд + сабмит
  features.py       add_features() — сюда ваш feature engineering; TablePrep — типы, категории, даты
  models.py         единый интерфейс моделей; если бустинг не установился — fallback на sklearn
  metrics.py        roc_auc, gini, f1, rmse, ... + подбор порога
  inference.py      predict_df() — одна функция для API, Streamlit и скрипта
  text_baseline.py  TF-IDF (слова + символы) + LogReg для текстов на рус/каз
eda/quick_eda.py    HTML-отчёт: пропуски, распределения, корреляции, adversarial validation
notebooks/          01_baseline.ipynb — то же самое в ноутбуке + анализ ошибок
app/                Streamlit-демо: скоринг файла, ручной ввод, порог, важность признаков
api/                FastAPI: /health, /model_info, /predict, /predict_file
scripts/            check_env.py, make_fake_data.py
reports/            eda_report.html, feature_importance.csv, experiments.csv (журнал всех запусков)
```

## Ключевые флаги `src.train`

| Флаг | Когда использовать |
|---|---|
| `--metric gini` | Метрику **всегда** берите ту же, что в лидерборде. Банки часто используют Gini = 2·AUC−1 |
| `--group client_id` | Несколько строк на одного клиента — иначе CV будет завышен (утечка через клиента) |
| `--time-col date` | Test «из будущего» — валидация по времени вместо случайного KFold |
| `--drop col1 col2` | Убрать утечки / мусор (id-подобные, «будущие» признаки) |
| `--models lgb,cb` | Выбор моделей; `hgb,linear` работают вообще без бустингов |
| `--tag fe_v2` | Подпись эксперимента в `reports/experiments.csv` |

## Что поменять под свой кейс

1. `src/features.py → add_features()` — ваши фичи (они автоматически применятся и в API, и в демо).
2. `app/streamlit_app.py` — заголовок, описание бизнес-проблемы, объяснение решения (SHAP).
3. `README.md` — описание решения для жюри (как запустить, архитектура, метрики).

См. также [CHECKLIST.md](CHECKLIST.md) — план подготовки и советы.
