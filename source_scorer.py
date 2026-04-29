"""
source_scorer.py
----------------
Haber kaynağının domain'ine göre güvenilirlik skoru döner.
Statik bir veritabanı kullanır; dış API gerektirmez.

Skor sistemi:
    9–10 : Çok güvenilir (büyük ana akım medya, resmi kaynaklar)
    7–8  : Güvenilir (köklü gazeteler, ajanslar)
    5–6  : Orta (partizan ama gazetecilik standartlarına uyan)
    3–4  : Düşük (sarı basın, doğrulanmamış iddialar)
    1–2  : Çok düşük (dezenformasyon / propaganda)
"""

from __future__ import annotations

from typing import TypedDict


class SourceInfo(TypedDict):
    score: int           # 1-10
    category: str        # "Ana Akım" | "Bağımsız" | "Sarı Basın" | "Ajans" | "Resmi" | "Uluslararası"
    bias: str            # "Sol" | "Sağ" | "Merkez" | "Belirsiz"
    note: str            # Kısa not


# ─── Türk ve uluslararası haber kaynakları veritabanı ────────────────────────

_SOURCE_DB: dict[str, SourceInfo] = {

    # ── Haber Ajansları ────────────────────────────────────────────
    "aa.com.tr": {
        "score": 7, "category": "Ajans", "bias": "Merkez",
        "note": "Anadolu Ajansı — Türkiye'nin resmi haber ajansı. Ulusal konularda hükümet yakın yayın yapabilir.",
    },
    "dha.com.tr": {
        "score": 7, "category": "Ajans", "bias": "Merkez",
        "note": "Demirören Haber Ajansı — Geniş kapsamlı, ajans haberciliği.",
    },
    "iha.com.tr": {
        "score": 7, "category": "Ajans", "bias": "Merkez",
        "note": "İhlas Haber Ajansı — Geniş kapsamlı haber ajansı.",
    },
    "bianet.org": {
        "score": 8, "category": "Bağımsız", "bias": "Sol",
        "note": "Bağımsız gazetecilik. Basın özgürlüğü ve azınlık hakları konularında kapsamlı.",
    },

    # ── Ana Akım Gazeteler ─────────────────────────────────────────
    "hurriyet.com.tr": {
        "score": 7, "category": "Ana Akım", "bias": "Merkez",
        "note": "Hürriyet — Türkiye'nin en köklü gazetelerinden.",
    },
    "milliyet.com.tr": {
        "score": 7, "category": "Ana Akım", "bias": "Merkez",
        "note": "Milliyet — Uzun geçmişe sahip ulusal gazete.",
    },
    "sabah.com.tr": {
        "score": 6, "category": "Ana Akım", "bias": "Sağ",
        "note": "Sabah — Geniş okuyucu kitlesine sahip; hükümete yakın yayın çizgisi.",
    },
    "cumhuriyet.com.tr": {
        "score": 7, "category": "Ana Akım", "bias": "Sol",
        "note": "Cumhuriyet — Türkiye'nin en eski gazetelerinden; muhalif çizgi.",
    },
    "haberturk.com": {
        "score": 7, "category": "Ana Akım", "bias": "Merkez",
        "note": "Habertürk — TV kanalı ve internet haberciliği.",
    },
    "ntv.com.tr": {
        "score": 7, "category": "Ana Akım", "bias": "Merkez",
        "note": "NTV — Köklü haber kanalı.",
    },
    "cnnturk.com": {
        "score": 7, "category": "Ana Akım", "bias": "Merkez",
        "note": "CNN Türk — Uluslararası CNN franchise'ı.",
    },
    "sozcu.com.tr": {
        "score": 6, "category": "Ana Akım", "bias": "Sol",
        "note": "Sözcü — Muhalif, milliyetçi sol çizgi.",
    },
    "posta.com.tr": {
        "score": 5, "category": "Ana Akım", "bias": "Merkez",
        "note": "Posta — Popüler, magazin ağırlıklı.",
    },
    "star.com.tr": {
        "score": 5, "category": "Ana Akım", "bias": "Sağ",
        "note": "Star — Hükümete yakın yayın politikası.",
    },
    "yenisafak.com": {
        "score": 5, "category": "Ana Akım", "bias": "Sağ",
        "note": "Yeni Şafak — Muhafazakâr-İslamcı çizgi.",
    },
    "turkiyegazetesi.com.tr": {
        "score": 5, "category": "Ana Akım", "bias": "Sağ",
        "note": "Türkiye Gazetesi — Muhafazakâr yayın.",
    },
    "takvim.com.tr": {
        "score": 4, "category": "Sarı Basın", "bias": "Sağ",
        "note": "Takvim — Magazin ve sarı basın eğilimi; abartılı başlıklar.",
    },
    "aksam.com.tr": {
        "score": 5, "category": "Ana Akım", "bias": "Sağ",
        "note": "Akşam — Milliyetçi çizgi.",
    },
    "gazeteduvar.com.tr": {
        "score": 7, "category": "Bağımsız", "bias": "Sol",
        "note": "Gazete Duvar — Bağımsız dijital yayın organı.",
    },
    "t24.com.tr": {
        "score": 7, "category": "Bağımsız", "bias": "Sol",
        "note": "T24 — Muhalif dijital haber platformu.",
    },
    "diken.com.tr": {
        "score": 6, "category": "Bağımsız", "bias": "Sol",
        "note": "Diken — Bağımsız, eleştirel haber.",
    },
    "birgun.net": {
        "score": 6, "category": "Bağımsız", "bias": "Sol",
        "note": "BirGün — Sol demokratik çizgi.",
    },
    "evrensel.net": {
        "score": 6, "category": "Bağımsız", "bias": "Sol",
        "note": "Evrensel — Emek ve sol odaklı.",
    },
    "medyascope.tv": {
        "score": 8, "category": "Bağımsız", "bias": "Merkez",
        "note": "Medyascope — Bağımsız, araştırmacı gazetecilik.",
    },
    "gazeteoksijen.com": {
        "score": 6, "category": "Bağımsız", "bias": "Merkez",
        "note": "Oksijen — Araştırmacı habercilik.",
    },

    # ── Spor ──────────────────────────────────────────────────────
    "sporx.com": {
        "score": 7, "category": "Ana Akım", "bias": "Belirsiz",
        "note": "Sporx — Spor haberciliğinde güvenilir.",
    },
    "fanatik.com.tr": {
        "score": 6, "category": "Ana Akım", "bias": "Belirsiz",
        "note": "Fanatik — Spor gazetesi.",
    },
    "fotomac.com.tr": {
        "score": 6, "category": "Ana Akım", "bias": "Belirsiz",
        "note": "Fotomaç — Spor odaklı yayın.",
    },

    # ── Resmi ve Akademik ─────────────────────────────────────────
    "tccb.gov.tr": {
        "score": 9, "category": "Resmi", "bias": "Merkez",
        "note": "Türkiye Cumhurbaşkanlığı resmi sitesi.",
    },
    "tbmm.gov.tr": {
        "score": 9, "category": "Resmi", "bias": "Merkez",
        "note": "Türkiye Büyük Millet Meclisi resmi sitesi.",
    },
    "saglik.gov.tr": {
        "score": 9, "category": "Resmi", "bias": "Merkez",
        "note": "Türkiye Sağlık Bakanlığı.",
    },
    "tuik.gov.tr": {
        "score": 9, "category": "Resmi", "bias": "Merkez",
        "note": "Türkiye İstatistik Kurumu.",
    },

    # ── Uluslararası ──────────────────────────────────────────────
    "bbc.com": {
        "score": 9, "category": "Uluslararası", "bias": "Merkez",
        "note": "BBC — Küresel standartlarda gazetecilik.",
    },
    "bbc.co.uk": {
        "score": 9, "category": "Uluslararası", "bias": "Merkez",
        "note": "BBC — Küresel standartlarda gazetecilik.",
    },
    "reuters.com": {
        "score": 10, "category": "Uluslararası", "bias": "Merkez",
        "note": "Reuters — Dünyanın en güvenilir haber ajanslarından.",
    },
    "apnews.com": {
        "score": 10, "category": "Uluslararası", "bias": "Merkez",
        "note": "Associated Press — Küresel haber standardı.",
    },
    "theguardian.com": {
        "score": 9, "category": "Uluslararası", "bias": "Sol",
        "note": "The Guardian — Araştırmacı gazetecilik.",
    },
    "nytimes.com": {
        "score": 9, "category": "Uluslararası", "bias": "Sol",
        "note": "New York Times — Küresel referans gazete.",
    },
    "aljazeera.com": {
        "score": 7, "category": "Uluslararası", "bias": "Merkez",
        "note": "Al Jazeera — Geniş uluslararası kapsam; Körfez etkisi.",
    },
    "dw.com": {
        "score": 8, "category": "Uluslararası", "bias": "Merkez",
        "note": "Deutsche Welle — Alman devlet yayın kuruluşu.",
    },

    # ── Düşük Güvenilirlik ────────────────────────────────────────
    "odatv.com": {
        "score": 4, "category": "Sarı Basın", "bias": "Sol",
        "note": "Oda TV — Sık sık doğrulanmamış iddialar.",
    },
    "haber7.com": {
        "score": 4, "category": "Sarı Basın", "bias": "Sağ",
        "note": "Haber7 — Clickbait başlıklar, abartılı haberler.",
    },
    "internethaber.com": {
        "score": 3, "category": "Sarı Basın", "bias": "Belirsiz",
        "note": "İnternet Haber — Doğrulama zayıf, tıklama odaklı.",
    },
}


def score_source(domain: str) -> dict:
    """
    Verilen domain için güvenilirlik bilgisi döner.

    Parameters
    ----------
    domain : str
        Kaynak domain (örn. 'sabah.com.tr').

    Returns
    -------
    dict
        {
            "known": bool,
            "score": int,           # 1–10
            "score_pct": int,       # 0–100 (görselleştirme için)
            "category": str,
            "bias": str,
            "note": str,
            "color": str,           # CSS rengi
            "emoji": str,           # Hızlı görsel
        }
    """
    domain = domain.lower().strip()

    # 'www.' ön ekini kaldır
    if domain.startswith("www."):
        domain = domain[4:]

    info = _SOURCE_DB.get(domain)

    if info is None:
        return {
            "known": False,
            "score": 0,
            "score_pct": 0,
            "category": "Bilinmiyor",
            "bias": "Belirsiz",
            "note": "Bu kaynak veritabanımızda bulunmuyor. Haberi bağımsız kaynaklardan doğrulayın.",
            "color": "#808080",
            "emoji": "❓",
        }

    score = info["score"]
    return {
        "known": True,
        "score": score,
        "score_pct": score * 10,
        "category": info["category"],
        "bias": info["bias"],
        "note": info["note"],
        "color": _score_to_color(score),
        "emoji": _score_to_emoji(score),
    }


def _score_to_color(score: int) -> str:
    if score >= 9:
        return "#00c851"
    elif score >= 7:
        return "#7bcf7b"
    elif score >= 5:
        return "#ffd700"
    elif score >= 3:
        return "#ff8c00"
    else:
        return "#ff4b4b"


def _score_to_emoji(score: int) -> str:
    if score >= 9:
        return "✅"
    elif score >= 7:
        return "🟢"
    elif score >= 5:
        return "🟡"
    elif score >= 3:
        return "🟠"
    else:
        return "🔴"


def get_all_sources() -> dict[str, SourceInfo]:
    """Tüm kayıtlı kaynakları döner (geliştirici/debug için)."""
    return dict(_SOURCE_DB)
