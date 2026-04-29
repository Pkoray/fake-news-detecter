<div align="center">

# 🔎 Büyüteç Haber Ajansı

### Yapay Zeka Destekli Türkçe Sahte Haber Tespit Sistemi

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> Makine öğrenmesi ve doğal dil işleme teknikleriyle haber metinlerini gerçek zamanlı analiz eden, açıklanabilir ve kaynak doğrulamalı bir sahte haber dedektörü.

![Demo Screenshot](assets/screenshot.png)

</div>

---

## 📋 İçindekiler

- [✨ Özellikler](#-özellikler)
- [🏗️ Proje Yapısı](#-proje-yapısı)
- [⚙️ Kurulum](#-kurulum)
- [🚀 Kullanım](#-kullanım)
- [🤖 Model Eğitimi](#-model-eğitimi)
- [🧩 Modüller](#-modüller)
- [📊 Model Performansı](#-model-performansı)
- [🛠️ Teknolojiler](#-teknolojiler)
- [🤝 Katkı Sağlama](#-katkı-sağlama)

---

## ✨ Özellikler

| Özellik | Açıklama |
|---|---|
| ✍️ **Metin Analizi** | Haber metnini yapıştır, anında sahte/gerçek tahmini al |
| 🔗 **URL Analizi** | Haber linkini gir, makale otomatik çekilsin ve analiz edilsin |
| ⭐ **Kaynak Güvenilirlik Skoru** | 50+ Türk ve uluslararası kaynak için 1-10 güvenilirlik puanı |
| 🔍 **Açıklanabilirlik** | Hangi kelimeler kararı tetikledi? Görsel kelime etki analizi |
| 📋 **Geçmiş Takibi** | Tüm analizler SQLite'ta kaydedilir, filtreleme ve CSV export |
| 📊 **Toplu Analiz** | Birden fazla haberi aynı anda analiz et |
| 🎯 **Risk Seviyeleri** | Çok Yüksek → Çok Düşük arasında 5 kademeli risk skoru |

---

## 🏗️ Proje Yapısı

```
fake-news-detecter/
│
├── app.py                          # Ana Streamlit uygulaması
├── requirements.txt                # Python bağımlılıkları
├── README.md
│
├── src/                            # Kaynak modüller
│   ├── predict.py                  # Tahmin motoru (ana API)
│   ├── preprocess.py               # Metin ön işleme (tokenize, stopword, stemming)
│   ├── train_model.py              # Model eğitim scripti
│   ├── create_turkish_datasets.py  # Türkçe veri seti oluşturucu
│   ├── url_fetcher.py              # URL'den makale metni çekici
│   ├── source_scorer.py            # Kaynak güvenilirlik veritabanı
│   ├── explainer.py                # Kelime etki analizi (açıklanabilirlik)
│   └── history.py                  # SQLite geçmiş yönetimi
│
├── model/                          # Eğitilmiş model dosyaları
│   ├── model.pkl                   # Logistic Regression / Random Forest modeli
│   └── vectorizer.pkl              # TF-IDF vektörleştirici
│
└── data/                           # Veri setleri
    ├── dataset.csv                 # Ana eğitim verisi (Fake + Real birleşik)
    ├── Fake.csv                    # Sahte haber örnekleri
    ├── True.csv                    # Gerçek haber örnekleri
    ├── fnd_turkish.csv             # Türkçe sahte haber verisi
    ├── liar_turkish.csv            # LIAR dataset Türkçe uyarlaması
    └── history.db                  # Analiz geçmişi (SQLite, otomatik oluşur)
```

---

## ⚙️ Kurulum

### Gereksinimler
- Python **3.10** veya üzeri
- pip

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/KULLANICI_ADINIZ/fake-news-detecter.git
cd fake-news-detecter
```

### 2. Sanal Ortam Oluşturun

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 4. NLTK Verilerini İndirin

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 🚀 Kullanım

### Streamlit Web Arayüzü (Önerilen)

```bash
streamlit run app.py
```

Tarayıcıda otomatik açılır → **http://localhost:8501**

### Komut Satırı (CLI)

```bash
# Tek metin analizi
python src/predict.py "Hükümet içme suyuna zihin kontrol kimyasalı karıştırıyor!"

# Çıktı:
# ═══════════════════════════════════════════════════════
#    FAKE NEWS DETECTOR — TAHMİN
# ═══════════════════════════════════════════════════════
#   Metin  : Hükümet içme suyuna zihin kontrol kimyasalı...
#   Sonuç  : 🔴 FAKE NEWS
#   Güven  : %94.3
#   Sahte  : %94.3
#   Gerçek : %5.7
#   Risk   : Çok Yüksek Risk
# ═══════════════════════════════════════════════════════
```

---

## 🤖 Model Eğitimi

> ⚠️ **Not:** `model/` klasöründeki `.pkl` dosyaları büyük olduğu için GitHub'a yüklenmez (`.gitignore`). Modeli kendiniz eğitmeniz gerekir.

### Hızlı Başlangıç (Demo Veri — ~30 saniye)

```bash
python src/train_model.py --demo
```

Gerçek bir veri seti olmadan 1.000 örnek sentetik veriyle hızlıca eğitir.

### Logistic Regression ile Eğitim (Varsayılan, ~98% doğruluk)

```bash
python src/train_model.py --model lr
```

### Random Forest ile Eğitim

```bash
python src/train_model.py --model rf
```

### Türkçe Veri Seti Oluşturma

```bash
python src/create_turkish_datasets.py
```

Eğitim tamamlandıktan sonra `model/model.pkl` ve `model/vectorizer.pkl` oluşturulur.

---

## 🧩 Modüller

### `src/predict.py` — Tahmin Motoru

```python
from src.predict import get_prediction_details

result = get_prediction_details("Haber metni buraya...")
print(result["label"])       # 'FAKE NEWS' veya 'REAL NEWS'
print(result["confidence"])  # Güven skoru (0-100)
print(result["risk_level"])  # Risk seviyesi
```

### `src/url_fetcher.py` — URL Makale Çekici

```python
from src.url_fetcher import fetch_article, extract_domain

result = fetch_article("https://www.bbc.com/turkce/articles/...")
print(result["text"])        # Makale metni
print(result["word_count"])  # Kelime sayısı
print(result["domain"])      # 'bbc.com'
```

### `src/source_scorer.py` — Kaynak Güvenilirlik Skoru

```python
from src.source_scorer import score_source

info = score_source("bbc.com")
print(info["score"])     # 9
print(info["category"])  # 'Uluslararası'
print(info["bias"])      # 'Merkez'
print(info["note"])      # Açıklayıcı not
```

Veritabanındaki kaynaklar:

| Kategori | Örnekler |
|---|---|
| 🌐 Uluslararası | bbc.com, reuters.com, apnews.com, nytimes.com |
| 📰 Ana Akım | hurriyet.com.tr, cumhuriyet.com.tr, ntv.com.tr |
| 🏛️ Resmi | tccb.gov.tr, tbmm.gov.tr, tuik.gov.tr |
| 📡 Ajans | aa.com.tr, dha.com.tr, iha.com.tr |
| 🔓 Bağımsız | t24.com.tr, bianet.org, medyascope.tv |
| ⚠️ Sarı Basın | takvim.com.tr, odatv.com, haber7.com |

### `src/explainer.py` — Açıklanabilirlik

```python
from src.explainer import get_text_word_scores

result = get_text_word_scores(metin, model_dir="model/", top_n=10)
for w in result["word_scores"]:
    print(w["word"], w["direction"], w["impact"])
```

### `src/history.py` — Geçmiş Takibi

```python
from src.history import save_analysis, get_history, get_stats, export_to_csv

# Analiz kaydet
save_analysis(text="...", result="FAKE", confidence=91.5)

# Geçmişi getir
rows = get_history(filter_result="FAKE", search_query="ekonomi")

# İstatistikler
stats = get_stats()
print(stats["total"], stats["fake_pct"])

# CSV dışa aktar
csv_string = export_to_csv()
```

---

## 📊 Model Performansı

| Metrik | Logistic Regression | Random Forest |
|---|---|---|
| **Doğruluk** | ~%98 | ~%97 |
| **Precision** | ~%98 | ~%97 |
| **Recall** | ~%98 | ~%97 |
| **F1 Score** | ~%98 | ~%97 |
| **Eğitim Süresi** | ~2 dakika | ~8 dakika |

> 📌 Değerler, `data/dataset.csv` (Fake + True birleşik, ~40K örnek) üzerinde %80 eğitim / %20 test bölünmesiyle elde edilmiştir.

### TF-IDF Konfigürasyonu

```
max_features = 50.000
ngram_range  = (1, 2)   # unigram + bigram
min_df       = 2
sublinear_tf = True
```

---

## 🛠️ Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| [Python 3.10+](https://python.org) | Temel programlama dili |
| [Streamlit](https://streamlit.io) | Web arayüzü |
| [Scikit-Learn](https://scikit-learn.org) | ML modelleri (LR, RF, TF-IDF) |
| [NLTK](https://nltk.org) | Doğal dil işleme (stopword, tokenizer) |
| [Pandas / NumPy](https://pandas.pydata.org) | Veri işleme |
| [Joblib](https://joblib.readthedocs.io) | Model serileştirme |
| [Requests](https://requests.readthedocs.io) | URL makale çekme |
| [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) | HTML parse |
| [SQLite3](https://docs.python.org/3/library/sqlite3.html) | Geçmiş veritabanı |

---

## 📁 GitHub'a Yüklenmeyen Dosyalar

Aşağıdaki dosyalar `.gitignore` ile hariç tutulmuştur. Bunları kendiniz oluşturmanız gerekir:

| Dosya | Boyut | Nasıl Oluşturulur |
|---|---|---|
| `model/model.pkl` | ~400 KB | `python src/train_model.py` |
| `model/vectorizer.pkl` | ~2 MB | `python src/train_model.py` |
| `data/dataset.csv` | ~107 MB | `data/Fake.csv` + `data/True.csv` birleşimi |
| `data/Fake.csv` | ~60 MB | [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| `data/True.csv` | ~51 MB | [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| `data/history.db` | Otomatik | İlk çalıştırmada oluşur |

---

## 🤝 Katkı Sağlama

Pull request'ler memnuniyetle karşılanır!

```bash
# Fork'layın ve kendi branch'inizi oluşturun
git checkout -b feature/yeni-ozellik

# Değişikliklerinizi commit edin
git commit -m "feat: yeni özellik eklendi"

# Branch'inizi push edin
git push origin feature/yeni-ozellik

# Pull Request açın
```

---

## ⚠️ Sorumluluk Reddi

Bu sistem **yardımcı bir araçtır**. Sonuçlar kesin doğru olmayabilir. Kritik haberleri mutlaka birden fazla güvenilir kaynaktan doğrulayın. Bu araç, basın özgürlüğü ve sansür amacıyla kullanılamaz.

---

<div align="center">

Geliştirici ile iletişim için [GitHub Issues](../../issues) açabilirsiniz.

⭐ Projeyi beğendiyseniz star atmayı unutmayın!

</div>
