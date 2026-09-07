from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import load_predictor, score_feature_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_mlp.pt"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

app = FastAPI(title="Fraud Model Scorer", version="1.0.0")
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR), name="dashboard")


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    account_age_days: float = Field(ge=0)
    transactions_24h: int = Field(ge=0)
    distance_km: float = Field(ge=0)
    card_present: int = Field(ge=0, le=1)


class ScoreResponse(BaseModel):
    fraud_probability: float
    review_recommended: bool
    threshold: float


@lru_cache
def predictor():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train --demo --epochs 30` first.")
    return load_predictor(MODEL_PATH)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"model_available": MODEL_PATH.exists()}


@app.post("/score", response_model=ScoreResponse)
def score(transaction: Transaction) -> ScoreResponse:
    try:
        artifact, model = predictor()
        probability = float(score_feature_rows(artifact, model, [transaction.model_dump()])[0])
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ScoreResponse(
        fraud_probability=probability,
        review_recommended=probability >= float(artifact["threshold"]),
        threshold=float(artifact["threshold"]),
    )
