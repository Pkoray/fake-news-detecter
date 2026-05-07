"""FastAPI backend for fake-news-detecter inference."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict import predict

app = FastAPI(title="Fake News Detecter API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    try:
        label, fake_prob, real_prob = predict(req.text)
        confidence = max(fake_prob, real_prob)
        normalized_label = "FAKE" if "FAKE" in label else "REAL"
        return PredictResponse(label=normalized_label, confidence=confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
