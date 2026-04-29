"""
app.py
------
Fake News Detector — Streamlit Web Arayüzü
Kullanıcının haber metni girmesine, analiz etmesine
ve sonucu görsel olarak incelemesine olanak tanır.

Çalıştırma:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import time

# Kaynak modülleri import et
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import get_prediction_details, predict_batch
from url_fetcher import fetch_article, extract_domain, check_dependencies as url_deps_ok
from source_scorer import score_source
from explainer import get_word_importance, get_text_word_scores
from history import init_db, save_analysis, get_history, get_stats, delete_record, clear_all, export_to_csv

# Geçmiş DB'yi başlat
_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "history.db")
init_db(_DB_PATH)

# ──────────────────────────────────────────────
# SAYFA AYARLARI
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Büyüteç Haber Ajansı",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS STİLLERİ
# ──────────────────────────────────────────────

st.markdown("""
<style>
/* Ana arka plan */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Başlık */
.main-title {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}

.sub-title {
    text-align: center;
    color: #a0a0c0;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Sonuç kartları */
.result-card-fake {
    background: linear-gradient(135deg, #ff4b4b22, #ff000011);
    border: 2px solid #ff4b4b;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}

.result-card-real {
    background: linear-gradient(135deg, #00c85122, #00ff5511);
    border: 2px solid #00c851;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,75,75,0.4); }
    50%       { box-shadow: 0 0 20px 8px rgba(255,75,75,0.15); }
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,200,81,0.4); }
    50%       { box-shadow: 0 0 20px 8px rgba(0,200,81,0.15); }
}

.result-label {
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}

.result-label-fake { color: #ff4b4b; }
.result-label-real { color: #00c851; }

.confidence-text {
    color: #e0e0f0;
    font-size: 1.05rem;
}

/* Metric kartları */
.metric-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1rem 0;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.8rem 1.5rem;
    text-align: center;
    min-width: 120px;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #7b2ff7;
}

.metric-label {
    font-size: 0.75rem;
    color: #909090;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* Bilgi kutusu */
.info-box {
    background: rgba(123,47,247,0.12);
    border-left: 4px solid #7b2ff7;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #c0c0e0;
    font-size: 0.9rem;
    margin: 1rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    color: #e0e0f0 !important;
    border: 1px solid rgba(123,47,247,0.4) !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
}

.stTextArea textarea:focus {
    border-color: #7b2ff7 !important;
    box-shadow: 0 0 0 2px rgba(123,47,247,0.25) !important;
}

/* Buton */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(123,47,247,0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(123,47,247,0.6) !important;
}

/* Örnek haberler */
.example-news {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: #b0b0d0;
    cursor: pointer;
}

/* Progress bar rengi */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7b2ff7, #00d2ff) !important;
}

/* Sekme */
.stTabs [data-baseweb="tab"] {
    color: #a0a0c0;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #7b2ff7 !important;
    border-bottom-color: #7b2ff7 !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.1) !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────

def render_result_card(details: dict) -> None:
    """
    Tahmin sonucunu görsel kart olarak render eder.

    Parameters
    ----------
    details : dict
        get_prediction_details() fonksiyonundan dönen sözlük.
    """
    is_fake = details["is_fake"]
    card_cls = "result-card-fake" if is_fake else "result-card-real"
    label_cls = "result-label-fake" if is_fake else "result-label-real"

    # Risk rengi
    risk_colors = {
        "Çok Yüksek Risk": "#ff4b4b",
        "Yüksek Risk":     "#ff8c00",
        "Orta Risk":       "#ffd700",
        "Düşük Risk":      "#90ee90",
        "Çok Düşük Risk":  "#00c851",
    }
    risk_color = risk_colors.get(details["risk_level"], "#ffffff")

    st.markdown(f"""
    <div class="{card_cls}">
        <div class="result-label {label_cls}">
            {details["label"]}
        </div>
        <div class="confidence-text">
            Güven Skoru: <strong>%{details["confidence"]:.1f}</strong>
        </div>
        <div>
            <span class="risk-badge" style="background:{risk_color}22; color:{risk_color}; border:1px solid {risk_color};">
                {details["risk_level"]}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(details: dict) -> None:
    """
    Sahte/Gerçek olasılıklarını progress bar ile gösterir.

    Parameters
    ----------
    details : dict
        Tahmin detayları sözlüğü.
    """
    st.markdown("#### Olasılık Dağılımı")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sahte Haber**")
        st.progress(details["fake_probability"] / 100)
        st.markdown(f"<center><b>%{details['fake_probability']:.1f}</b></center>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Gerçek Haber**")
        st.progress(details["real_probability"] / 100)
        st.markdown(f"<center><b>%{details['real_probability']:.1f}</b></center>", unsafe_allow_html=True)


def render_text_stats(details: dict, original_text: str) -> None:
    """
    Analiz edilen metin hakkındaki istatistikleri gösterir.

    Parameters
    ----------
    details : dict
        Tahmin detayları sözlüğü.
    original_text : str
        Orijinal metin.
    """
    st.markdown("#### Metin İstatistikleri")
    m1, m2, m3 = st.columns(3)
    m1.metric("Karakter Sayısı", len(original_text))
    m2.metric("Kelime Sayısı", details["word_count"])
    m3.metric("İşlenen Kelime", details["processed_words"])


def render_sidebar() -> None:
    """
    Sidebar içeriğini (hakkında, nasıl çalışır, örnek haberler) oluşturur.
    """
    with st.sidebar:
        st.markdown("## Büyüteç Haber Ajansı")
        st.markdown("---")

        st.markdown("### Nasıl Çalışır?")
        st.markdown("""
        <div class="info-box">
        1. Haber metnini metin kutusuna girin<br>
        2. <b>Analiz Et</b> butonuna tıklayın<br>
        3. Sonucu ve olasılıkları inceleyin
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Model Hakkında")
        st.markdown("""
        <div class="info-box">
        • <b>Algoritma:</b> Logistic Regression / Random Forest<br>
        • <b>Özellik:</b> TF-IDF (50K feature, bigram)<br>
        • <b>Veri:</b> Fake & Real News Dataset<br>
        • <b>Doğruluk:</b> ~98% (gerçek veriyle)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Uyarı")
        st.info(
            "Bu sistem yardımcı bir araçtır. "
            "Sonuçlar kesin doğru olmayabilir. "
            "Kritik haberleri mutlaka birden fazla "
            "güvenilir kaynaktan doğrulayın."
        )

        st.markdown("### Teknolojiler")
        techs = ["Python 3.10+", "Scikit-Learn", "NLTK", "TF-IDF", "Streamlit", "Joblib"]
        for t in techs:
            st.markdown(f"• {t}")

        st.markdown("---")
        st.caption("© 2026 Büyüteç Haber Ajansı")


# ──────────────────────────────────────────────
# ANA UYGULAMA
# ──────────────────────────────────────────────

def main() -> None:
    """
    Streamlit uygulamasının ana fonksiyonu.
    Tüm UI bileşenlerini oluşturur ve kullanıcı etkileşimini yönetir.
    """

    # Sidebar
    render_sidebar()

    # Başlık
    st.markdown('<div class="main-title">Büyüteç Haber Ajansı</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Yapay Zeka Destekli Sahte Haber Tespit Sistemi</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Model durumu kontrolü — hem model/ klasöründe hem kök dizinde ara
    _base = os.path.dirname(__file__)
    model_path = os.path.join(_base, "model", "model.pkl")
    vectorizer_path = os.path.join(_base, "model", "vectorizer.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(_base, "model.pkl")
    if not os.path.exists(vectorizer_path):
        vectorizer_path = os.path.join(_base, "vectorizer.pkl")
    model_ready = os.path.exists(model_path) and os.path.exists(vectorizer_path)

    if not model_ready:
        st.warning(
            "Model henüz eğitilmemiş! "
            "Lütfen önce terminalde şu komutu çalıştırın:\n\n"
            "```bash\npython src/train_model.py --demo\n```\n\n"
            "Gerçek veri setiyle eğitmek için:\n"
            "```bash\npython src/train_model.py\n```"
        )

    # ── Sekmeler ──────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✍️ Metin Analizi",
        "🔗 URL Analizi",
        "📊 Toplu Analiz",
        "📋 Geçmiş",
        "⚙️ Sistem Bilgisi",
    ])

    # ──────────────────────────────────────────
    # SEKMEİ 1: TEK METİN ANALİZİ
    # ──────────────────────────────────────────
    with tab1:
        col_input, col_result = st.columns([1.2, 1], gap="large")

        with col_input:
            st.markdown("### Haber Metni Girin")

            # Örnek haberler
            st.markdown("##### Örnek Metinler")
            example_col1, example_col2 = st.columns(2)

            real_example = (
                "Merkez Bankası, enflasyonla mücadele kapsamında politika faizini 250 baz puan "
                "artırarak yüzde 45'e yükseltti. Para Politikası Kurulu, kararın oy birliğiyle "
                "alındığını açıkladı. Başkan, faiz kararının yıl sonuna kadar enflasyonu "
                "tek haneye indirmeyi hedeflediğini belirtti. Ekonomistler, kararın piyasalar "
                "tarafından büyük ölçüde beklendiğini ifade etti."
            )

            fake_example = (
                "ACİL UYARI: Hükümet, içme suyuna gizlice zihin kontrol kimyasalı karıştırıyor! "
                "Anonim bir içeriden kaynak, uzun süredir gizlenen bu gerçeği ortaya çıkardı. "
                "Ana akım medya bu haberi KASITLI olarak saklıyor. Silenmeden önce paylaşın! "
                "Küresel seçkinler halkı uyuşturmak için bu planı 20 yıldır uyguluyor."
            )

            with example_col1:
                if st.button("✅ Gerçek Haber Örneği", use_container_width=True):
                    st.session_state["news_input"] = real_example

            with example_col2:
                if st.button("❌ Sahte Haber Örneği", use_container_width=True):
                    st.session_state["news_input"] = fake_example

            # Metin alanı
            news_text = st.text_area(
                label="Haber metnini buraya yapıştırın veya yazın:",
                height=280,
                placeholder=(
                    "Buraya analiz etmek istediğiniz haber metnini girin...\n\n"
                    "Not: En iyi sonuçlar için en az 50 kelime içeren metinler kullanın."
                ),
                key="news_input",
            )

            # Karakter sayacı
            if news_text:
                char_count = len(news_text)
                word_count = len(news_text.split())
                st.caption(f"{char_count} karakter | {word_count} kelime")

            # Analiz butonu
            analyze_btn = st.button(
                "ANALİZ ET",
                disabled=not model_ready or not news_text.strip(),
                use_container_width=True,
            )

        with col_result:
            st.markdown("### Analiz Sonucu")

            if analyze_btn and news_text.strip():
                with st.spinner("Yapay zeka analiz ediyor..."):
                    time.sleep(0.5)
                    try:
                        details = get_prediction_details(news_text)
                        st.session_state["last_result"] = details
                        st.session_state["last_text"]   = news_text
                        # Geçmişe kaydet
                        save_analysis(
                            text=news_text,
                            result="FAKE" if details["is_fake"] else "REAL",
                            confidence=details["confidence"],
                            risk_level=details.get("risk_level", ""),
                            source_type="text",
                            db_path=_DB_PATH,
                        )
                    except FileNotFoundError as e:
                        st.error(f"Model bulunamadı!\n\n{e}")
                    except ValueError as e:
                        st.warning(f"{e}")
                    except Exception as e:
                        st.error(f"Beklenmeyen hata: {e}")

            # Sonucu göster
            if "last_result" in st.session_state:
                details = st.session_state["last_result"]
                original_text = st.session_state.get("last_text", "")

                # Sonuç kartı
                render_result_card(details)
                st.markdown("<br>", unsafe_allow_html=True)

                # Olasılık barları
                render_probability_bars(details)
                st.markdown("<br>", unsafe_allow_html=True)

                # Metin istatistikleri
                if original_text:
                    render_text_stats(details, original_text)

                # Açıklanabilirlik
                st.markdown("---")
                model_dir = os.path.join(os.path.dirname(__file__), "model")
                word_result = get_text_word_scores(original_text, model_dir, top_n=8)
                if word_result and word_result.get("success"):
                    with st.expander("🔍 Kelime Etkisi — Hangi kelimeler bu kararı tetikledi?", expanded=False):
                        ws = word_result["word_scores"]
                        fake_ws = [(w["word"], w["impact"]) for w in ws if w["direction"] == "sahte"]
                        real_ws = [(w["word"], w["impact"]) for w in ws if w["direction"] == "gerçek"]
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            st.markdown("**🔴 Sahte habere işaret edenler**")
                            for word, score in fake_ws[:5]:
                                bar = min(int(abs(score) * 30), 20) * "█"
                                st.markdown(f"`{word}` {bar} `{abs(score):.3f}`")
                        with ec2:
                            st.markdown("**🟢 Gerçek habere işaret edenler**")
                            for word, score in real_ws[:5]:
                                bar = min(int(abs(score) * 30), 20) * "█"
                                st.markdown(f"`{word}` {bar} `{abs(score):.3f}`")

                # Yorum
                st.markdown("---")
                if details["is_fake"]:
                    if details["fake_probability"] >= 85:
                        st.error(
                            "Bu metin **yüksek olasılıkla** sahte haber içeriyor. "
                            "Lütfen güvenilir kaynaklardan doğrulayın."
                        )
                    else:
                        st.warning(
                            "Bu metin **şüpheli** unsurlar içeriyor olabilir. "
                            "Dikkatli olun ve kaynağı kontrol edin."
                        )
                else:
                    if details["real_probability"] >= 85:
                        st.success(
                            "Bu metin **güvenilir** görünüyor. "
                            "Yine de önemli kararlar için kaynağı doğrulayın."
                        )
                    else:
                        st.info(
                            "Bu metin **büyük olasılıkla** gerçek, ancak orta düzey belirsizlik var."
                        )
            else:
                st.markdown("""
                <div style="
                    border: 2px dashed rgba(123,47,247,0.3);
                    border-radius: 16px;
                    padding: 3rem;
                    text-align: center;
                    color: #606090;
                ">
                    <div style="font-size: 1.1rem;">Henüz analiz yapılmadı</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                        Sol taraftaki alana haber metnini girin ve<br>
                        <b>Analiz Et</b> butonuna tıklayın
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────
    # SEKME 2: URL ANALİZİ
    # ──────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🔗 URL ile Haber Analizi")
        st.markdown("Haber linkini girin — makale otomatik çekilsin ve analiz edilsin.")

        if not url_deps_ok():
            st.warning(
                "⚠️ URL analizi için ek kütüphaneler gerekiyor. Lütfen şu komutu çalıştırın:\n\n"
                "```bash\npip install requests beautifulsoup4 lxml\n```"
            )
        else:
            url_input = st.text_input(
                "Haber URL'si:",
                placeholder="https://www.bbc.com/turkce/articles/...",
                key="url_input",
            )

            fetch_btn = st.button(
                "🔗 Makaleyi Çek ve Analiz Et",
                disabled=not model_ready or not url_input.strip(),
                use_container_width=True,
            )

            if fetch_btn and url_input.strip():
                with st.spinner("Makale çekiliyor..."):
                    fetch_result = fetch_article(url_input.strip())

                if not fetch_result["success"]:
                    st.error(f"❌ Makale çekilemedi: {fetch_result['error']}")
                else:
                    st.success(f"✅ Makale çekildi — {fetch_result['word_count']} kelime")
                    if fetch_result["title"]:
                        st.markdown(f"**Başlık:** {fetch_result['title']}")

                    with st.spinner("Analiz ediliyor..."):
                        time.sleep(0.3)
                        try:
                            url_details = get_prediction_details(fetch_result["text"])
                            # Geçmişe kaydet
                            save_analysis(
                                text=fetch_result["text"],
                                result="FAKE" if url_details["is_fake"] else "REAL",
                                confidence=url_details["confidence"],
                                risk_level=url_details.get("risk_level", ""),
                                url=url_input.strip(),
                                domain=fetch_result["domain"],
                                source_type="url",
                                db_path=_DB_PATH,
                            )

                            col_u1, col_u2 = st.columns([1, 1], gap="large")
                            with col_u1:
                                st.markdown("#### Analiz Sonucu")
                                render_result_card(url_details)
                                st.markdown("<br>", unsafe_allow_html=True)
                                render_probability_bars(url_details)

                            with col_u2:
                                # Kaynak skoru
                                st.markdown("#### ⭐ Kaynak Güvenilirlik Skoru")
                                domain = fetch_result["domain"]
                                src = score_source(domain)
                                if src["known"]:
                                    st.markdown(
                                        f"{src['emoji']} **{domain}** — "
                                        f"{src['category']} | Yanaşma: {src['bias']}"
                                    )
                                    st.progress(src["score_pct"] / 100)
                                    st.markdown(
                                        f"<b style='color:{src['color']}'>Skor: {src['score']}/10</b>",
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(src["note"])
                                else:
                                    st.warning(f"❓ `{domain}` — {src['note']}")

                                # Metin özeti
                                st.markdown("#### 📝 Makale Özeti")
                                snippet = fetch_result["text"][:500]
                                st.text_area("Makale metni (ilk 500 karakter):", snippet, height=150, disabled=True)

                        except Exception as e:
                            st.error(f"Analiz hatası: {e}")

    # ──────────────────────────────────────────
    # SEKME 3: TOPLU ANALİZ
    # ──────────────────────────────────────────
    with tab3:
        st.markdown("### Toplu Haber Analizi")
        st.markdown(
            "Her satıra bir haber metni girin. "
            "Sistem her birini ayrı ayrı analiz edecektir."
        )

        bulk_text = st.text_area(
            "Her satıra bir haber metni:",
            height=200,
            placeholder=(
                "1. Haber metni...\n"
                "2. Haber metni...\n"
                "3. Haber metni..."
            ),
        )

        bulk_btn = st.button(
            "TOPLU ANALİZ",
            disabled=not model_ready or not bulk_text.strip(),
            use_container_width=False,
        )

        if bulk_btn and bulk_text.strip():
            lines = [line.strip() for line in bulk_text.split("\n") if line.strip()]
            texts = []
            for line in lines:
                import re
                clean_line = re.sub(r"^\d+[\.\)]\s*", "", line)
                if len(clean_line) > 20:
                    texts.append(clean_line)

            if not texts:
                st.warning("Lütfen en az 20 karakterden oluşan geçerli metinler girin.")
            else:
                with st.spinner(f"{len(texts)} metin analiz ediliyor..."):
                    results = predict_batch(texts)

                st.markdown(f"#### Sonuçlar ({len(results)} metin)")

                for i, res in enumerate(results, 1):
                    if res.get("label") == "ERROR":
                        st.error(f"**{i}.** Hata: {res.get('error')}")
                        continue

                    is_fake = res["label"] == "FAKE NEWS"
                    icon = "SAHTE" if is_fake else "GERCEK"
                    conf = res["confidence"] * 100

                    with st.expander(
                        f"**{i}.** {res['text']} — **{res['label']}** (%{conf:.1f})"
                    ):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sonuç", res["label"])
                        c2.metric("Sahte Olasılığı", f"%{res['fake_probability']*100:.1f}")
                        c3.metric("Gerçek Olasılığı", f"%{res['real_probability']*100:.1f}")

    # ──────────────────────────────────────────
    # SEKME 4: GEÇMİŞ
    # ──────────────────────────────────────────
    with tab4:
        st.markdown("### 📋 Geçmiş Analizler")

        stats = get_stats(_DB_PATH)
        if stats["total"] == 0:
            st.info("💭 Henüz hiç analiz yapılmadı. Metin veya URL analizi yaptıktan sonra burada görünecek.")
        else:
            # İstatistik kartları
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("📊 Toplam Analiz", stats["total"])
            sc2.metric("❌ Sahte Haber", f"{stats['fake_count']} (%{stats['fake_pct']})")
            sc3.metric("✅ Gerçek Haber", f"{stats['real_count']} (%{stats['real_pct']})")
            sc4.metric("🎯 Ort. Güven", f"%{stats['avg_confidence']:.1f}")

            st.markdown("---")

            # Filtreler
            fcol1, fcol2, fcol3 = st.columns([1, 2, 1])
            with fcol1:
                filter_result = st.selectbox("Filtrele:", ["Tümü", "FAKE", "REAL"], key="hist_filter")
            with fcol2:
                search_q = st.text_input("🔍 Ara:", placeholder="metin veya domain...", key="hist_search")
            with fcol3:
                csv_data = export_to_csv(filter_result, search_q, _DB_PATH)
                st.download_button(
                    "⬇️ CSV İndir",
                    data=csv_data,
                    file_name="gecmis_analizler.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            rows = get_history(filter_result, search_q, limit=50, db_path=_DB_PATH)

            if not rows:
                st.warning("Bu filtreye uyan kayıt bulunamadı.")
            else:
                st.caption(f"{len(rows)} kayıt gösteriliyor (maks. 50)")
                for row in rows:
                    is_fake = row["result"] == "FAKE"
                    icon    = "❌" if is_fake else "✅"
                    label   = "SAHTE" if is_fake else "GERÇEK"
                    src_icon = "🔗" if row["source_type"] == "url" else "✍️"
                    with st.expander(
                        f"{icon} [{row['created_at']}] {src_icon} {row['text_snippet'][:80]}..."
                    ):
                        dc1, dc2, dc3 = st.columns(3)
                        dc1.markdown(f"**Sonuç:** `{label}`")
                        dc2.markdown(f"**Güven:** `%{row['confidence']:.1f}`")
                        dc3.markdown(f"**Risk:** `{row['risk_level'] or '-'}`")
                        if row["url"]:
                            st.markdown(f"🔗 [{row['url']}]({row['url']})")
                        if row["domain"]:
                            src = score_source(row["domain"])
                            st.markdown(
                                f"{src['emoji']} Kaynak: **{row['domain']}** — Skor: {src['score']}/10 ({src['category']})"
                            )
                        if st.button(f"🗑️ Sil", key=f"del_{row['id']}"):
                            delete_record(row["id"], _DB_PATH)
                            st.rerun()

            st.markdown("---")
            if st.button("⚠️ Tüm Geçmişi Temizle", type="secondary"):
                clear_all(_DB_PATH)
                st.success("Geçmiş temizlendi.")
                st.rerun()

    # ──────────────────────────────────────────
    # SEKME 5: SİSTEM BİLGİSİ
    # ──────────────────────────────────────────
    with tab5:
        st.markdown("### Sistem ve Model Bilgisi")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Dosya Durumu")
            files_to_check = {
                "model/model.pkl":       "Model dosyası",
                "model/vectorizer.pkl":  "Vektörleştirici",
                "data/dataset.csv":      "Veri seti",
                "src/preprocess.py":     "Ön işleme modülü",
                "src/train_model.py":    "Eğitim modülü",
                "src/predict.py":        "Tahmin modülü",
            }
            for fpath, fname in files_to_check.items():
                full_path = os.path.join(os.path.dirname(__file__), fpath)
                exists = os.path.exists(full_path)
                icon = "✅" if exists else "❌"
                size = ""
                if exists:
                    size_bytes = os.path.getsize(full_path)
                    if size_bytes > 1_000_000:
                        size = f" ({size_bytes/1_000_000:.1f} MB)"
                    elif size_bytes > 1_000:
                        size = f" ({size_bytes/1_000:.1f} KB)"
                st.markdown(f"{icon} **{fname}**{size}")

        with col_b:
            st.markdown("#### Eğitim Komutu")
            st.code(
                "# Demo veriyle hızlı başlangıç\n"
                "python src/train_model.py --demo\n\n"
                "# Logistic Regression (varsayılan)\n"
                "python src/train_model.py --model lr\n\n"
                "# Random Forest\n"
                "python src/train_model.py --model rf",
                language="bash",
            )

            st.markdown("#### Arayüzü Başlatma")
            st.code("streamlit run app.py", language="bash")

            st.markdown("#### Gereksinimleri Yükleme")
            st.code("pip install -r requirements.txt", language="bash")

        st.markdown("---")
        st.markdown("#### Kullanılan Teknolojiler")

        tech_data = {
            "Python 3.10+":          "Temel programlama dili",
            "Scikit-Learn":          "ML modelleri ve değerlendirme",
            "NLTK":                  "Doğal dil işleme (stopwords, tokenizer)",
            "TF-IDF Vectorizer":     "Metin özellik çıkarımı",
            "Logistic Regression":   "Birincil sınıflandırma modeli",
            "Random Forest":         "Alternatif ensemble modeli",
            "Joblib":                "Model serileştirme",
            "Streamlit":             "Web arayüzü",
            "Pandas / NumPy":        "Veri işleme",
        }

        cols = st.columns(3)
        for i, (tech, desc) in enumerate(tech_data.items()):
            with cols[i % 3]:
                st.markdown(f"**{tech}**")
                st.caption(desc)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    main()
