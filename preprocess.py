"""
preprocess.py
-------------
Haber metinleri için NLP ön işleme modülü.
Türkçe ve İngilizce ikidilli destek sunar.
Lowercase, noktalama temizleme, stopword kaldırma,
tokenization ve TF-IDF vektörleştirme işlemlerini içerir.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Gerekli NLTK verilerini indir
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── TÜRKÇE STOPWORD LİSTESİ ──────────────────────────────────────────────────
TURKISH_STOPWORDS = {
    "ve", "ile", "bir", "bu", "da", "de", "mi", "mu", "mü", "mı",
    "için", "ama", "ya", "ki", "ne", "en", "çok", "daha", "hem",
    "veya", "ancak", "fakat", "lakin", "oysa", "halbuki",
    "şu", "biz", "siz", "onlar", "ben", "sen", "benim",
    "senin", "onun", "bizim", "sizin", "onların", "bana", "sana",
    "ona", "bize", "size", "onlara", "bende", "sende", "onda",
    "bizde", "sizde", "onlarda", "benden", "senden", "ondan",
    "olan", "olarak", "oldu", "olur", "olması", "olduğu",
    "olacak", "edildi", "edilir", "edilmesi", "edilecek",
    "var", "yok", "gibi", "kadar", "göre", "karşı", "doğru",
    "önce", "sonra", "içinde", "dışında", "üzerinde", "altında",
    "yanında", "arasında", "üzerine", "altına", "içine", "dışına",
    "hakkında", "tarafından", "itibaren", "beri", "rağmen",
    "üzere", "dolayı", "nedeniyle", "yüzünden", "sayesinde",
    "böyle", "şöyle", "öyle", "nasıl", "neden", "niçin",
    "nerede", "nereden", "nereye", "hangi", "kim", "kimin",
    "her", "hiç", "bazı", "birkaç", "bütün", "tüm", "hepsi",
    "çeşitli", "farklı", "diğer", "sadece", "yalnızca",
    "bile", "dahi", "zaten", "artık", "henüz", "hala", "pek",
    "oldukça", "fazla", "az", "biraz", "asla", "kesinlikle",
    "mutlaka", "elbette", "tabii", "ise", "diye", "diyerek",
    "şey", "şeyi", "şeyde", "şeye", "şeyden", "şeyler",
    "kez", "kere", "defa", "sefer", "yıl", "ay", "gün", "saat",
    "söyledi", "belirtti", "açıkladı", "ifade", "etti", "dedi",
    "konuştu", "aktardı", "vurguladı", "paylaştı", "yer", "yere",
    "olup", "üst", "alt", "ön", "arka", "iç", "dış", "yan",
    "bunun", "şunun", "bunlar", "şunlar", "bunları", "şunları",
    "yapılan", "yapılacak", "yapılmış", "yapıyor", "yapıldı",
    "verilen", "verilecek", "verilmiş", "veriyor", "verildi",
    "alınan", "alınacak", "alınmış", "alıyor", "alındı",
    "olduğunu", "olduğu", "olduğunda", "olduğundan",
    "olacağını", "olacağı", "olmadığını", "olmadığı",
    "ettiğini", "ettiği", "edeceğini", "edeceği",
    "çünkü", "zira", "nitekim", "özellikle", "ayrıca",
    "amacıyla", "kapsamında", "çerçevesinde", "sürecinde",
    "konusunda", "durumunda", "noktasında", "açısından",
    "üzerinden", "yoluyla", "vasıtasıyla", "aracılığıyla",
    "sonucunda", "ardından", "öncesinde", "sonrasında",
    "birlikte", "beraber", "beraberinde", "karşılıklı",
    "olduğu", "olduğunu", "olduklarını", "olmaktadır",
    "edilmektedir", "bulunmaktadır", "gelmektedir",
    "görmektedir", "almaktadır", "vermektedir",
}

# İngilizce stopword'ler
_en_stops = set(stopwords.words("english"))

# Birleşik Türkçe + İngilizce stopword seti
STOP_WORDS = TURKISH_STOPWORDS | _en_stops

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # proje kökü
VECTORIZER_PATH = os.path.join(_BASE_DIR, "model", "vectorizer.pkl")
if not os.path.exists(VECTORIZER_PATH):
    VECTORIZER_PATH = os.path.join(_BASE_DIR, "vectorizer.pkl")


def clean_text(text: str) -> str:
    """
    Ham metni temizler: küçük harfe çevirir, noktalama ve
    sayıları kaldırır. Türkçe karakterleri (ğ, ü, ş, ı, ö, ç) korur.

    Parameters
    ----------
    text : str
        Temizlenecek ham metin.

    Returns
    -------
    str
        Temizlenmiş metin.
    """
    if not isinstance(text, str):
        return ""

    # Küçük harfe çevir
    text = text.lower()

    # URL'leri kaldır
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # HTML tag'lerini kaldır
    text = re.sub(r"<.*?>", "", text)

    # Noktalama kaldır — Türkçe harfleri koru
    text = re.sub(r"[^\w\sğüşıöçğüşıöç]", " ", text, flags=re.UNICODE)

    # Sayıları kaldır
    text = re.sub(r"\d+", "", text)

    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """
    Metinden Türkçe ve İngilizce stopword'leri kaldırır.

    Parameters
    ----------
    text : str
        Stopword'lerden arındırılacak metin.

    Returns
    -------
    str
        Stopword'leri kaldırılmış metin.
    """
    tokens = text.split()
    filtered = [
        word for word in tokens
        if word not in STOP_WORDS and len(word) > 2
    ]
    return " ".join(filtered)


def preprocess_text(text: str) -> str:
    """
    Tam NLP ön işleme pipeline'ını uygular:
    clean_text → remove_stopwords
    Türkçe ve İngilizce metinler için çalışır.

    Parameters
    ----------
    text : str
        İşlenecek ham haber metni.

    Returns
    -------
    str
        Tamamen işlenmiş metin.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
    """
    DataFrame içindeki metin sütununa ön işleme uygular.

    Parameters
    ----------
    df : pd.DataFrame
        İşlenecek veri çerçevesi.
    text_col : str
        Metin sütununun adı.

    Returns
    -------
    pd.Series
        İşlenmiş metin serisi.
    """
    print(f"[INFO] {len(df)} satır için metin ön işleme uygulanıyor...")
    processed = df[text_col].fillna("").apply(preprocess_text)
    print("[INFO] Ön işleme tamamlandı.")
    return processed


def build_tfidf_vectorizer(max_features: int = 50000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    TF-IDF vektörleştirici oluşturur.
    Türkçe karakterleri destekleyen token_pattern kullanır.

    Parameters
    ----------
    max_features : int
        Maksimum özellik (kelime) sayısı.
    ngram_range : tuple
        Unigram ve bigram için (1, 2).

    Returns
    -------
    TfidfVectorizer
        Yapılandırılmış TF-IDF vektörleştirici.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )


def fit_and_save_vectorizer(texts: pd.Series, vectorizer: TfidfVectorizer) -> TfidfVectorizer:
    """
    TF-IDF vektörleştiricisini eğitim verisi üzerinde fit eder ve kaydeder.

    Parameters
    ----------
    texts : pd.Series
        Fit edilecek metin serisi.
    vectorizer : TfidfVectorizer
        Fit edilecek vektörleştirici.

    Returns
    -------
    TfidfVectorizer
        Fit edilmiş vektörleştirici.
    """
    print("[INFO] TF-IDF vektörleştirici eğitiliyor...")
    vectorizer.fit(texts)

    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[INFO] Vektörleştirici kaydedildi: {VECTORIZER_PATH}")
    return vectorizer


def load_vectorizer() -> TfidfVectorizer:
    """
    Kaydedilmiş TF-IDF vektörleştiricisini yükler.

    Returns
    -------
    TfidfVectorizer
        Yüklenmiş vektörleştirici.

    Raises
    ------
    FileNotFoundError
        Vektörleştirici dosyası bulunamazsa.
    """
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vektörleştirici bulunamadı: {VECTORIZER_PATH}\n"
            "Lütfen önce 'python src/train_model.py' komutunu çalıştırın."
        )
    return joblib.load(VECTORIZER_PATH)
