"""
predict.py
----------
Tahmin modülü.
Eğitilmiş model ve vektörleştirici kullanılarak
yeni haber metinleri üzerinde tahmin yapar.
"""

import os
import sys
from typing import Tuple
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_text, load_vectorizer

# Model yolu — hem model/ klasöründe hem kök dizinde ara
_BASE = os.path.dirname(os.path.dirname(__file__))  # proje kökü
_MODEL_PATH_DEFAULT = os.path.join(_BASE, "model", "model.pkl")
if not os.path.exists(_MODEL_PATH_DEFAULT):
    _MODEL_PATH_DEFAULT = os.path.join(_BASE, "model.pkl")


def load_model(path: str = None):
    """
    Kaydedilmiş modeli yükler.
    """
    if path is None:
        path = _MODEL_PATH_DEFAULT
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bulunamadı: {path}\n"
            "Lütfen önce 'python src/train_model.py' komutunu çalıştırın."
        )
    return joblib.load(path)

# Label haritası
LABEL_MAP = {0: "FAKE NEWS", 1: "REAL NEWS"}
LABEL_EMOJI = {0: "🔴", 1: "🟢"}
LABEL_COLOR = {0: "fake", 1: "real"}


def predict_news(text: str) -> Tuple[str, float, float]:
    """
    Verilen haber metninin gerçek mi sahte mi olduğunu tahmin eder.

    Parameters
    ----------
    text : str
        Tahmin edilecek haber metni.

    Returns
    -------
    Tuple[str, float, float]
        (label, fake_probability, real_probability)
        label: 'FAKE NEWS' veya 'REAL NEWS'
        fake_probability: Sahte olma olasılığı (0-1)
        real_probability: Gerçek olma olasılığı (0-1)

    Raises
    ------
    ValueError
        Metin boş veya çok kısaysa.
    """
    if not text or not text.strip():
        raise ValueError("Metin boş olamaz!")

    if len(text.strip()) < 20:
        raise ValueError("Metin çok kısa! En az 20 karakter girin.")

    # Model ve vektörleştiriciyi yükle
    model = load_model()
    vectorizer = load_vectorizer()

    # Ön işleme
    processed = preprocess_text(text)

    # TF-IDF dönüşümü
    text_tfidf = vectorizer.transform([processed])

    # Tahmin
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    fake_prob = probabilities[0]
    real_prob = probabilities[1]
    label = LABEL_MAP[int(prediction)]

    return label, float(fake_prob), float(real_prob)


def predict_batch(texts: list) -> list:
    """
    Birden fazla haber metni için toplu tahmin yapar.

    Parameters
    ----------
    texts : list
        Tahmin edilecek metin listesi.

    Returns
    -------
    list
        Her metin için (label, fake_prob, real_prob) tuple'larından oluşan liste.
    """
    if not texts:
        return []

    model = load_model()
    vectorizer = load_vectorizer()

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
    """
    Tahmin sonucu ile birlikte ek detayları döndürür.

    Parameters
    ----------
    text : str
        Analiz edilecek haber metni.

    Returns
    -------
    dict
        label, fake_prob, real_prob, confidence, risk_level
        ve işlenmiş metin bilgilerini içeren sözlük.
    """
    label, fake_prob, real_prob = predict_news(text)

    confidence = max(fake_prob, real_prob)

    # Risk seviyesi
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
    word_count_original  = len(text.split())
    word_count_processed = len(processed_text.split())

    return {
        "label":          label,
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "confidence":     round(confidence * 100, 2),
        "risk_level":     risk_level,
        "is_fake":        label == "FAKE NEWS",
        "word_count":     word_count_original,
        "processed_words": word_count_processed,
        "emoji":          LABEL_EMOJI[0 if label == "FAKE NEWS" else 1],
    }


# ──────────────────────────────────────────────
# MAIN (CLI kullanımı için)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake News Detector — Tahmin Aracı")
    parser.add_argument("text", type=str, help="Analiz edilecek haber metni")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("   FAKE NEWS DETECTOR — TAHMİN")
    print("="*55)

    try:
        details = get_prediction_details(args.text)
        print(f"\n  Metin  : {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
        print(f"\n  Sonuç  : {details['emoji']} {details['label']}")
        print(f"  Güven  : %{details['confidence']:.1f}")
        print(f"  Sahte  : %{details['fake_probability']:.1f}")
        print(f"  Gerçek : %{details['real_probability']:.1f}")
        print(f"  Risk   : {details['risk_level']}")
        print("\n" + "="*55)
    except Exception as e:
        print(f"\n[HATA] {e}\n")
