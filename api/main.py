"""REST API модели.  Запуск:  uvicorn api.main:app --reload --port 8000
Документация (Swagger) автоматически: http://localhost:8000/docs  — её удобно показывать жюри.

Примеры:
  curl localhost:8000/health
  curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
       -d '{"records":[{"age":35,"income":450000,"loan_amount":2000000,"region":"Астана"}]}'
  curl -X POST localhost:8000/predict_file -F "file=@data/raw/test.csv"
"""
import io
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import DEFAULT_BUNDLE, load_bundle, model_info, predict_df  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    if DEFAULT_BUNDLE.exists():
        load_bundle()  # грузим модель один раз при старте, а не на каждый запрос
    yield


app = FastAPI(title="Forte Hackathon Model API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=10_000,
                                          examples=[[{"age": 35, "income": 450000, "region": "Астана"}]])


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": DEFAULT_BUNDLE.exists()}


@app.get("/model_info")
def info():
    _require_model()
    return model_info()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _require_model()
    t0 = time.perf_counter()
    try:
        res = predict_df(pd.DataFrame(req.records))
    except Exception as e:  # отдаём понятную ошибку вместо 500
        raise HTTPException(422, f"Не удалось обработать данные: {e}")
    cols = [c for c in res.columns if c == "prediction" or c == "probability" or c.startswith("proba_")]
    return PredictResponse(predictions=_jsonable(res[cols]), latency_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    _require_model()
    raw = await file.read()
    name = (file.filename or "").lower()
    df = pd.read_excel(io.BytesIO(raw)) if name.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(raw))
    res = predict_df(df)
    return {"n": len(res), "predictions": _jsonable(res)}


def _require_model():
    if not DEFAULT_BUNDLE.exists():
        raise HTTPException(503, "Модель не обучена: запустите python -m src.train ...")


def _jsonable(df: pd.DataFrame) -> list[dict]:
    # через JSON: numpy-типы и NaN гарантированно превращаются в нормальные int/float/null
    return json.loads(df.to_json(orient="records", force_ascii=False, date_format="iso"))
