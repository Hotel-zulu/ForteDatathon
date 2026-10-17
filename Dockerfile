FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# ffmpeg — декодирование .mov (HEVC/H.264); libgl/libglib — для OpenCV; libgomp — для torch/onnx
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt
# Раскомментируйте, когда подключите модели (для GPU — базовый образ pytorch/pytorch:*-cuda*)
# RUN pip install -r requirements-ml.txt
COPY src ./src
COPY api ./api
COPY app ./app
COPY configs ./configs
COPY models ./models
# Офлайн-режим: веса уже лежат в models/, интернет при обработке не нужен (п. 7 ТЗ)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
HEALTHCHECK CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
