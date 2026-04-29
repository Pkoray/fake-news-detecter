"""
explainer.py
------------
Logistic Regression modelini kullanarak hangi kelimelerin
sahte / gerçek sınıflandırmasını en çok tetiklediğini gösterir.

eli5 kütüphanesi YOK ise alternatif olarak sklearn'in kendi
coefficient'larından doğrudan kelime önemleri çıkarılır.
Bu sayede eli5 bağımlılığı zorunlu değildir.
"""

from __future__ import annotations

import os
import joblib
import numpy as np


# ─── Yardımcı ────────────────────────────────────────────────────────────────

def _load_model_and_vectorizer(model_dir: str) -> tuple | None:
    """
    Model ve vectorizer dosyalarını yükler.
    Başarısız olursa None döner.

    Returns
    -------
    tuple (model, vectorizer) veya None
    """
    model_path      = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        return None

    try:
        model      = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception:
        return None


def _get_top_words_from_lr(model, vectorizer, top_n: int = 15) -> dict:
    """
    LogisticRegression modelinden en etkili kelimeleri çıkarır.

    Returns
    -------
    dict
        {
            "fake_words":  [(word, score), ...],
            "real_words":  [(word, score), ...],
            "method": "logistic_regression"
        }
    """
    # Tüm feature isimlerini al
    feature_names = np.array(vectorizer.get_feature_names_out())

    # LR'nin coefficient'larını al (binary: class 1 = REAL)
    coef = model.coef_[0]  # shape: (n_features,)

    # Pozitif coef → REAL habere işaret eder
    # Negatif coef → FAKE habere işaret eder
    top_fake_idx = np.argsort(coef)[:top_n]          # En negatif = Fake
    top_real_idx = np.argsort(coef)[::-1][:top_n]    # En pozitif = Real

    fake_words = [(feature_names[i], float(-coef[i])) for i in top_fake_idx]
    real_words = [(feature_names[i], float(coef[i]))  for i in top_real_idx]

    return {
        "fake_words": fake_words,
        "real_words": real_words,
        "method": "logistic_regression",
    }


def _get_top_words_from_rf(model, vectorizer, top_n: int = 15) -> dict:
    """
    RandomForest modelinden feature importance'a göre kelimeleri çıkarır.
    RF'de yön (sahte/gerçek) ayırt edilemez; en önemli kelimeler döner.

    Returns
    -------
    dict
        {
            "top_words":  [(word, score), ...],
            "method": "random_forest"
        }
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances   = model.feature_importances_

    top_idx   = np.argsort(importances)[::-1][:top_n]
    top_words = [(feature_names[i], float(importances[i])) for i in top_idx]

    return {
        "top_words": top_words,
        "method": "random_forest",
    }


def get_word_importance(model_dir: str, top_n: int = 10) -> dict | None:
    """
    Yüklü modele göre kelime önemlerini döner.

    Parameters
    ----------
    model_dir : str
        model.pkl ve vectorizer.pkl'ın bulunduğu dizin.
    top_n : int
        Her kategoriden kaç kelime gösterileceği.

    Returns
    -------
    dict veya None
        Başarılıysa:
            {
                "success": True,
                "method":  str,
                "fake_words": [(word, score), ...] | None,
                "real_words": [(word, score), ...] | None,
                "top_words":  [(word, score), ...] | None,  # RF için
            }
        Başarısızsa:
            {"success": False, "error": str}
    """
    loaded = _load_model_and_vectorizer(model_dir)
    if loaded is None:
        return {"success": False, "error": "Model veya vectorizer bulunamadı."}

    model, vectorizer = loaded

    # hasattr ile model tipini belirle
    model_class = type(model).__name__

    try:
        if hasattr(model, "coef_"):
            # LogisticRegression
            result = _get_top_words_from_lr(model, vectorizer, top_n)
            return {"success": True, **result}

        elif hasattr(model, "feature_importances_"):
            # RandomForest veya benzeri ensemble
            result = _get_top_words_from_rf(model, vectorizer, top_n)
            return {"success": True, **result}

        else:
            return {
                "success": False,
                "error": f"'{model_class}' modeli için kelime önemi desteklenmiyor.",
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_text_word_scores(text: str, model_dir: str, top_n: int = 10) -> dict | None:
    """
    Verilen metin içindeki kelimelerin modeldeki ağırlıklarını döner.
    Hangi kelimeler metinde geçiyor ve bunlar ne kadar etkili?

    Parameters
    ----------
    text : str
        Analiz edilecek metin.
    model_dir : str
        model.pkl ve vectorizer.pkl dizini.
    top_n : int
        Sonuç listesindeki kelime sayısı.

    Returns
    -------
    dict veya None
    """
    loaded = _load_model_and_vectorizer(model_dir)
    if loaded is None:
        return None

    model, vectorizer = loaded

    if not hasattr(model, "coef_"):
        return None  # Sadece LR destekleniyor

    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        coef          = model.coef_[0]

        # Metni transform et
        vec = vectorizer.transform([text])
        nonzero_indices = vec.nonzero()[1]

        word_scores = []
        for idx in nonzero_indices:
            word  = feature_names[idx]
            score = coef[idx]
            tf    = float(vec[0, idx])
            word_scores.append({
                "word":      word,
                "coef":      float(score),
                "tf":        tf,
                "impact":    float(score * tf),
                "direction": "gerçek" if score > 0 else "sahte",
            })

        # En etkilileri sırala
        word_scores.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return {
            "success":    True,
            "word_scores": word_scores[:top_n],
            "method":     "text_specific",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
