"""
predict.py
----------
Tahmin modülü (yalnızca inference).
Eğitilmiş model ve vektörleştiriciyi yükler,
yeni haber metinleri üzerinde tahmin yapar.
"""

import os
from typing import Tuple

import joblib

from preprocess import preprocess_text

_BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_BASE, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(_BASE, "model", "vectorizer.pkl")

# Label haritası
LABEL_MAP = {0: "FAKE NEWS", 1: "REAL NEWS"}
LABEL_EMOJI = {0: "🔴", 1: "🟢"}


def load_model():
    """Inference artefact'larını yükler, yoksa hata verir."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError("Model not found. Please run training during deployment.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str) -> Tuple[str, float, float]:
    """Tek bir metin için tahmin döndürür: (label, fake_prob, real_prob)."""
    if not text or not text.strip():
        raise ValueError("Metin boş olamaz!")
    if len(text.strip()) < 20:
        raise ValueError("Metin çok kısa! En az 20 karakter girin.")

    model, vectorizer = load_model()
    processed = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed])

    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    return LABEL_MAP[int(prediction)], float(probabilities[0]), float(probabilities[1])


def predict_batch(texts: list) -> list:
    """Birden fazla haber metni için toplu tahmin yapar."""
    if not texts:
        return []

    model, vectorizer = load_model()
    results = []
    for text in texts:
        try:
            processed = preprocess_text(text)
            text_tfidf = vectorizer.transform([processed])
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "label": LABEL_MAP[int(prediction)],
                "fake_probability": float(probabilities[0]),
                "real_probability": float(probabilities[1]),
                "confidence": float(max(probabilities)),
            })
        except Exception as e:
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "label": "ERROR",
                "error": str(e),
            })
    return results


def get_prediction_details(text: str) -> dict:
    """Tahmin sonucu ile birlikte ek detayları döndürür."""
    label, fake_prob, real_prob = predict(text)
    confidence = max(fake_prob, real_prob)

    if fake_prob >= 0.85:
        risk_level = "Çok Yüksek Risk"
    elif fake_prob >= 0.65:
        risk_level = "Yüksek Risk"
    elif fake_prob >= 0.45:
        risk_level = "Orta Risk"
    elif fake_prob >= 0.25:
        risk_level = "Düşük Risk"
    else:
        risk_level = "Çok Düşük Risk"

    processed_text = preprocess_text(text)
    return {
        "label": label,
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "confidence": round(confidence * 100, 2),
        "risk_level": risk_level,
        "is_fake": label == "FAKE NEWS",
        "word_count": len(text.split()),
        "processed_words": len(processed_text.split()),
        "emoji": LABEL_EMOJI[0 if label == "FAKE NEWS" else 1],
    }
