# Шорткаты. На Windows без make — просто копируйте команды.
TRAIN ?= data/raw/train.csv
TEST  ?= data/raw/test.csv
TARGET ?= target
ID ?= id

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt -r requirements-extra.txt

fake:
	python scripts/make_fake_data.py

eda:
	python eda/quick_eda.py $(TRAIN) --target $(TARGET) --test $(TEST) --id $(ID)

train:
	python -m src.train --train $(TRAIN) --test $(TEST) --target $(TARGET) --id $(ID)

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn api.main:app --reload --port 8000

docker:
	docker compose up --build

check:
	python scripts/check_env.py

smoke: fake eda train
	@echo "✅ Весь пайплайн отработал на синтетике"
