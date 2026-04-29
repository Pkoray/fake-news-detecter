"""
url_fetcher.py
--------------
Verilen URL'den makale metnini çeker.
requests + BeautifulSoup4 kullanır; dış API gerektirmez.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# İsteğe bağlı bağımlılıklar — yüklü değilse açıklayıcı hata döner
try:
    import requests
    from bs4 import BeautifulSoup
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


# Makale içeriği aramada kullanılan HTML tagları (öncelik sırasıyla)
_ARTICLE_TAGS = ["article", "main", "section"]
_FALLBACK_TAGS = ["div"]

# İçerikten çıkarılacak gürültülü CSS sınıfları / kimlikleri
_NOISE_CLASSES = {
    "nav", "navigation", "navbar", "menu", "footer", "header",
    "sidebar", "advertisement", "ad", "ads", "social", "share",
    "comment", "comments", "related", "trending", "newsletter",
}

# Tarayıcı user-agent (bazı siteler bot engelini atlatmak için)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
}


def check_dependencies() -> bool:
    """requests ve beautifulsoup4 kurulu mu?"""
    return _DEPS_OK


def extract_domain(url: str) -> str:
    """URL'den domain'i döner (örn. 'sabah.com.tr')."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # 'www.' ön ekini kaldır
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def _is_noisy_element(tag) -> bool:
    """
    Verilen BeautifulSoup tag'inin navigasyon/reklam/yorum gibi
    gürültülü bir element olup olmadığını kontrol eder.
    """
    classes = set(" ".join(tag.get("class", [])).lower().split())
    tag_id   = tag.get("id", "").lower()
    return bool(classes & _NOISE_CLASSES) or any(n in tag_id for n in _NOISE_CLASSES)


def _extract_text_from_soup(soup: "BeautifulSoup") -> str:
    """
    BeautifulSoup nesnesinden anlamlı makale metnini çıkarır.
    Önce <article>/<main>/<section> dener, bulamazsa <p> taglarına düşer.
    """
    # 1) Öncelikli containerlar
    for tag_name in _ARTICLE_TAGS:
        containers = soup.find_all(tag_name)
        for container in containers:
            if _is_noisy_element(container):
                continue
            paragraphs = container.find_all("p")
            text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
            if len(text.split()) > 50:  # Yeterince uzun mu?
                return _clean_text(text)

    # 2) Fallback: tüm <p> tagları
    all_p = soup.find_all("p")
    text = " ".join(p.get_text(separator=" ", strip=True) for p in all_p)
    if len(text.split()) > 20:
        return _clean_text(text)

    # 3) Son çare: body metni
    body = soup.find("body")
    if body:
        return _clean_text(body.get_text(separator=" ", strip=True))

    return ""


def _clean_text(text: str) -> str:
    """Fazla boşlukları ve özel karakterleri temizler."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:()\-–—\"'«»]", " ", text)
    return text.strip()


def fetch_article(url: str, timeout: int = 10) -> dict:
    """
    Verilen URL'yi çeker ve makale metnini döner.

    Parameters
    ----------
    url : str
        Analiz edilecek haber URL'si.
    timeout : int
        HTTP isteği için zaman aşımı (saniye).

    Returns
    -------
    dict
        {
            "success": bool,
            "text": str,          # Çekilen makale metni
            "title": str,         # Sayfa başlığı
            "domain": str,        # Kaynak domain
            "word_count": int,
            "error": str | None   # Hata varsa açıklama
        }
    """
    result = {
        "success": False,
        "text": "",
        "title": "",
        "domain": extract_domain(url),
        "word_count": 0,
        "error": None,
    }

    if not _DEPS_OK:
        result["error"] = (
            "'requests' ve 'beautifulsoup4' kütüphaneleri yüklü değil. "
            "Lütfen `pip install requests beautifulsoup4 lxml` çalıştırın."
        )
        return result

    if not url.startswith(("http://", "https://")):
        result["error"] = "Geçersiz URL. 'http://' veya 'https://' ile başlamalıdır."
        return result

    try:
        response = requests.get(url, headers=_HEADERS, timeout=timeout)
        response.raise_for_status()

        # Encoding düzelt
        if response.encoding and response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "lxml")

        # Gereksiz tagleri kaldır
        for tag in soup(["script", "style", "noscript", "iframe", "svg", "img"]):
            tag.decompose()

        # Başlık
        title_tag = soup.find("title")
        result["title"] = title_tag.get_text(strip=True) if title_tag else ""

        # Makale metni
        text = _extract_text_from_soup(soup)
        result["text"] = text
        result["word_count"] = len(text.split())
        result["success"] = bool(text and result["word_count"] > 10)

        if not result["success"]:
            result["error"] = (
                "Sayfa metni çıkarılamadı. "
                "Site dinamik içerik (JavaScript) kullanıyor olabilir."
            )

    except requests.exceptions.Timeout:
        result["error"] = f"Zaman aşımı: Site {timeout} saniye içinde yanıt vermedi."
    except requests.exceptions.ConnectionError:
        result["error"] = "Bağlantı hatası: URL'ye erişilemiyor."
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP hatası: {e.response.status_code} — {e.response.reason}"
    except Exception as e:
        result["error"] = f"Beklenmeyen hata: {str(e)}"

    return result
