FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# libgomp1 нужен LightGBM/XGBoost (без него ошибка "libgomp.so.1: cannot open shared object file")
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY api ./api
COPY app ./app
COPY models ./models

EXPOSE 8000 8501
HEALTHCHECK CMD curl -fs http://localhost:8000/health || exit 1
# По умолчанию поднимаем API. Для Streamlit: docker run ... streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
