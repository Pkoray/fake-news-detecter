"""
train_model.py
--------------
Model eğitim modülü.
Veriyi yükler, ön işler, TF-IDF uygular,
Logistic Regression + Random Forest eğitir,
değerlendirir ve modeli kaydeder.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Kendi modülümüzü import et
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import (
    preprocess_dataframe,
    build_tfidf_vectorizer,
    fit_and_save_vectorizer,
)

# Sabitler
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
RANDOM_SEED = 42


# ──────────────────────────────────────────────
# 1. VERİ YÜKLEME
# ──────────────────────────────────────────────

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    CSV veri setini yükler ve temel doğrulama yapar.

    Parameters
    ----------
    path : str
        Veri setinin dosya yolu.

    Returns
    -------
    pd.DataFrame
        Yüklenmiş veri çerçevesi.

    Raises
    ------
    FileNotFoundError
        Dosya bulunamazsa hata fırlatır.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Veri seti bulunamadı: {path}\n"
            "Lütfen Kaggle'dan 'Fake and Real News Dataset' indirip\n"
            "'data/dataset.csv' olarak kaydedin.\n"
            "Alternatif: python src/train_model.py --demo komutuyla demo veriyle çalışın."
        )

    print(f"[INFO] Veri seti yükleniyor: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Yüklenen satır sayısı: {len(df)}")
    print(f"[INFO] Sütunlar: {list(df.columns)}")
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham DataFrame'i model için hazırlar:
    - 'text' ve 'label' sütunlarını standartlaştırır
    - Eksik değerleri temizler
    - Label encoding uygular (REAL=1, FAKE=0)

    Parameters
    ----------
    df : pd.DataFrame
        Ham veri çerçevesi.

    Returns
    -------
    pd.DataFrame
        'text' ve 'label' sütunlarına sahip hazır DataFrame.
    """
    df = df.copy()

    # Sütun adlarını küçük harfe çevir
    df.columns = df.columns.str.lower().str.strip()

    # Metin sütununu belirle
    text_candidates = ["text", "content", "article", "body"]
    text_col = next((c for c in text_candidates if c in df.columns), None)

    if text_col is None:
        # İlk string sütunu kullan
        str_cols = [c for c in df.columns if df[c].dtype == object and c != "label"]
        if not str_cols:
            raise ValueError("Metin sütunu bulunamadı!")
        text_col = str_cols[0]
        print(f"[WARN] Metin sütunu otomatik seçildi: {text_col}")

    # Başlık varsa metne ekle
    if "title" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df[text_col].fillna("")
    else:
        df["text"] = df[text_col].fillna("")

    # Label sütununu belirle ve encode et
    label_candidates = ["label", "class", "fake", "type"]
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if label_col is None:
        raise ValueError("Label sütunu bulunamadı! Sütunlar: " + str(list(df.columns)))

    # Label değerlerini 0/1'e çevir
    unique_labels = df[label_col].unique()
    print(f"[INFO] Benzersiz label değerleri: {unique_labels}")

    if df[label_col].dtype == object:
        label_map = {}
        for val in unique_labels:
            val_str = str(val).strip().upper()
            if any(keyword in val_str for keyword in ["REAL", "TRUE", "LEGIT", "1"]):
                label_map[val] = 1
            else:
                label_map[val] = 0
        df["label"] = df[label_col].map(label_map)
        print(f"[INFO] Label mapping: {label_map}")
    else:
        df["label"] = df[label_col].astype(int)

    # Eksik değerleri temizle
    df = df[["text", "label"]].dropna()
    df = df[df["text"].str.strip() != ""]

    print(f"[INFO] Temizlenen satır sayısı: {len(df)}")
    print(f"[INFO] Sınıf dağılımı:\n{df['label'].value_counts()}")
    return df


# ──────────────────────────────────────────────
# 2. MODEL EĞİTİMİ
# ──────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Logistic Regression modelini eğitir.

    Parameters
    ----------
    X_train : sparse matrix
        Eğitim özellik matrisi.
    y_train : array-like
        Eğitim etiketleri.

    Returns
    -------
    LogisticRegression
        Eğitilmiş model.
    """
    print("[INFO] Logistic Regression eğitiliyor...")
    model = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[INFO] Logistic Regression eğitimi tamamlandı.")
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Random Forest modelini eğitir.

    Parameters
    ----------
    X_train : sparse matrix
        Eğitim özellik matrisi.
    y_train : array-like
        Eğitim etiketleri.

    Returns
    -------
    RandomForestClassifier
        Eğitilmiş model.
    """
    print("[INFO] Random Forest eğitiliyor (bu biraz sürebilir)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[INFO] Random Forest eğitimi tamamlandı.")
    return model


# ──────────────────────────────────────────────
# 3. DEĞERLENDİRME
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Modeli test seti üzerinde değerlendirir ve metrikleri raporlar.

    Parameters
    ----------
    model : sklearn estimator
        Değerlendirilecek eğitilmiş model.
    X_test : sparse matrix
        Test özellik matrisi.
    y_test : array-like
        Gerçek test etiketleri.
    model_name : str
        Raporda gösterilecek model adı.

    Returns
    -------
    dict
        accuracy, precision, recall, f1 değerlerini içeren sözlük.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n{'='*55}")
    print(f"  {model_name} — Değerlendirme Sonuçları")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"{'='*55}")
    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    return metrics


# ──────────────────────────────────────────────
# 4. MODEL KAYDETME
# ──────────────────────────────────────────────

def save_model(model, path: str = MODEL_PATH) -> None:
    """
    Eğitilmiş modeli joblib ile kaydeder.

    Parameters
    ----------
    model : sklearn estimator
        Kaydedilecek model.
    path : str
        Kaydedilecek dosya yolu.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n[INFO] Model kaydedildi: {path}")


def load_model(path: str = MODEL_PATH):
    """
    Kaydedilmiş modeli yükler.

    Parameters
    ----------
    path : str
        Model dosyasının yolu.

    Returns
    -------
    sklearn estimator
        Yüklenmiş model.

    Raises
    ------
    FileNotFoundError
        Model dosyası bulunamazsa hata fırlatır.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bulunamadı: {path}\n"
            "Lütfen önce 'python src/train_model.py' komutunu çalıştırın."
        )
    return joblib.load(path)


# ──────────────────────────────────────────────
# 5. DEMO VERİ OLUŞTURUCU
# ──────────────────────────────────────────────

def create_demo_dataset(path: str = DATA_PATH, n_samples: int = 1000) -> None:
    """
    Gerçek veri seti olmadığında Türkçe ve İngilizce
    ikidilli demo veri seti oluşturur.

    Parameters
    ----------
    path : str
        Kaydedilecek CSV dosya yolu.
    n_samples : int
        Her sınıf için oluşturulacak örnek sayısı.
    """
    np.random.seed(RANDOM_SEED)

    # ── TÜRKÇE GERÇEK HABERLER ──────────────────────────────────────────
    real_tr = [
        "Merkez Bankası Para Politikası Kurulu, politika faizini 250 baz puan artırarak yüzde 45 düzeyine yükseltti. Kurul kararının oy birliğiyle alındığı açıklandı. Başkan, söz konusu kararın enflasyonla mücadele sürecinde önemli bir adım olduğunu vurguladı. Ekonomistler kararın piyasalar tarafından büyük ölçüde beklendiğini belirtti.",
        "Sağlık Bakanlığı, yeni geliştirilen aşının klinik denemelerinde yüzde 94 etkinlik oranına ulaştığını duyurdu. Üç aşamadan oluşan klinik süreçte 12 bin gönüllü yer aldı. Aşının önümüzdeki altı ay içinde kamuya sunulması planlanıyor. Dünya Sağlık Örgütü süreci olumlu karşıladığını açıkladı.",
        "Belediye meclisi, toplu taşıma altyapısına yönelik 5 milyar liralık yatırım paketini onayladı. Proje kapsamında 50 kilometre yeni metro hattı ve 200 elektrikli otobüs filosu hayata geçirilecek. Çalışmaların üç yılda tamamlanması öngörülüyor. Uzmanlar projenin karbon salınımını yüzde 30 azaltacağını tahmin ediyor.",
        "Borsa İstanbul'da işlem gören teknoloji şirketi, son çeyrekte beklentilerin üzerinde kâr açıkladı. Gelirler bir önceki yılın aynı dönemine kıyasla yüzde 22 artış gösterdi. Yönetim kurulu temettü dağıtımına karar verdi. Şirket hisseleri haberin ardından yüzde 7 değer kazandı.",
        "Tarım Bakanlığı, organik tarım desteklerini bu yıldan itibaren iki katına çıkardığını bildirdi. Destek başvuruları önümüzdeki ay başlayacak olup 50 bin çiftçinin yararlanması bekleniyor. Uzmanlar adımın ülkenin tarım ihracatına olumlu yansıyacağını öngörüyor. Bakanlık ayrıca sulama altyapısı için ek kaynak tahsis edeceğini de açıkladı.",
        "Üniversite araştırmacıları, güneş enerjisi dönüşüm verimliliğini yüzde 40 artıran yeni bir panel teknolojisi geliştirdi. Sonuçlar uluslararası hakemli bir dergide yayımlandı. Teknolojinin ticarileşmesi halinde yenilenebilir enerji maliyetlerini önemli ölçüde düşürmesi bekleniyor. Çalışma için çeşitli uluslararası kuruluşlardan fon desteği alındı.",
        "Adalet Bakanlığı, mahkeme süreçlerini hızlandırmayı hedefleyen dijital dönüşüm paketini kamuoyuyla paylaştı. Paket kapsamında tüm belge işlemleri elektronik ortama taşınacak ve duruşmalar çevrimiçi gerçekleştirilebilecek. Pilot uygulamanın gelecek yıl başlaması planlanıyor. Bakanlar bu reformun dava çözüm sürelerini yarıya indireceğini ifade etti.",
        "Milli Eğitim Bakanlığı, dijital okuryazarlık eğitimini ilköğretimden itibaren müfredata zorunlu ders olarak ekledi. Yeni müfredat gelecek dönem hayata geçecek; 50 bin öğretmene yönelik mesleki gelişim programları da başlatılacak. Uzmanlar adımı eğitimdeki dijital uçurumun kapatılması açısından kritik buluyor. Program, avrupa standartlarıyla uyum içinde tasarlandı.",
        "Çevre ve Şehircilik Bakanlığı, kıyı temizliği için başlatılan kampanya kapsamında 500 ton atığın toplandığını açıkladı. 10 bin gönüllünün katılımıyla gerçekleştirilen etkinlik, ülke tarihinin en büyük kıyı temizliği kampanyası olma özelliğini taşıyor. Yetkililer bu tür kampanyaların yıllık düzenli etkinliğe dönüştürüleceğini bildirdi. Girişim uluslararası çevre kuruluşlarından da takdir gördü.",
        "Türkiye İstatistik Kurumu açıkladığı verilere göre işsizlik oranı geçen yılın aynı dönemine kıyasla 1,5 puan gerileyerek yüzde 9,2 düzeyine indi. İstihdam artışının ağırlıklı olarak hizmet ve teknoloji sektörlerinden kaynaklandığı görüldü. Ekonomistler olumlu eğilimin sürmesi için yapısal reformların hayata geçirilmesi gerektiğini vurguladı. Bakanlık ek istihdam teşvik programlarını devreye alacağını açıkladı.",
        "Büyükşehir belediyesi, akıllı şehir projesinin ilk fazını tamamladığını duyurdu. Proje kapsamında trafik yönetim sistemleri yapay zeka ile entegre edildi, 10 bin sokak lambası enerji tasarruflu modele dönüştürüldü. Yetkililer bu adımın yıllık enerji giderlerini yüzde 35 düşürdüğünü belirtti. İkinci fazın önümüzdeki yıl başlaması bekleniyor.",
        "Sanayi Bakanlığı, yerli elektrikli araç üretimine yönelik teşvik paketini açıkladı. Yerli üreticilere sağlanacak vergi indirimleri ve hibe desteklerinin sektöre 10 milyar liralık yatırım çekeceği öngörülüyor. Yetkililer 2030 yılına kadar araç satışlarının yüzde 30'unun elektrikli olmasını hedeflediklerini açıkladı. Proje aynı zamanda yeni istihdam alanları yaratacak.",
    ]

    # ── TÜRKÇE SAHTE HABERLER ───────────────────────────────────────────
    fake_tr = [
        "ACİL UYARI: Hükümet, içme suyuna gizlice zihin kontrol kimyasalı karıştırıyor! Anonim bir içeriden kaynak bu gizli gerçeği ifşa etti. Ana akım medya bu haberi KASITLI olarak saklıyor. Silenmeden önce acilen paylaşın! Küresel seçkinler halkı uyuşturmak için bu planı 20 yıldır uyguluyor ve kimse görmezden gelmenizi istiyor!",
        "BOMBA İDDİA: Gizli belgeler sızdırıldı, devlet vatandaşları aşı yoluyla mikroçiple kontrol etmeyi planlıyor! Hayatını riske atan anonim ihbarcı bu şok edici gerçeği dünyaya duyurdu. Küresel seçkinler bunu BİLMENİZİ istemiyor, bu yüzden medya tamamen SANSÜR uyguluyor. Silinmeden önce herkese iletin!",
        "ŞOKTAN DONACAKSINIZ: 5G kuleleri gerçekte insan beynini kontrol etmek için tasarlanmış gizli silahlar! Bunu kanıtlayan belgeler büyük teknoloji şirketleri tarafından kasıtlı olarak bastırılıyor. Anonim içeriden kaynaklar dünya liderlerinin 1990'dan beri bu planı gizli yeraltı sığınaklarında hazırladığını doğruladu. UYAKIN, çok geç olmadan her yerde paylaşın!",
        "BÜYÜK YALAN ORTAYA ÇIKTI: Büyük ilaç şirketleri onlarca yıldır kanser ilacını gizliyor! Basit bitkisel bir kombinasyon 72 saatte kanseri, diyabeti ve kalp hastalığını tamamen iyileştiriyor. Bu bulguları yayımlamaya çalışan doktorların lisansları yolsuzluk içindeki tıp kuruluşlarınca iptal edildi. Trilyon dolarlık ilaç endüstrisi bu sırrın çıkmasına ASLA izin vermez!",
        "GİZLİ GÜNDEM İFŞA OLDU: Gıdalara kasıtlı olarak kısırlaştırıcı maddeler ekleniyor, bu bir nüfus azaltma programının parçası! Sektörden sızdırılan iç yazışmalar yöneticilerin bunu on yıllardır bildiğini kanıtlıyor. Market rafları sizi ve ailenizi ZEHİRLİYOR. Yerel organik gıdayla beslenin yoksa kendi yıkımınıza ortak oluyorsunuz!",
        "MEDYANIN SAKLADIĞI GERÇEK: Seçim sonuçları oy makinelerine yerleştirilen gizli bir algoritmayla çalındı! Dahi istatistikçi ihbarcıların matematiksel analizi, oyların istatistiksel olarak imkânsız olduğunu yüzde 100 kesinlikle ispatlıyor. Dürüst hâkimler delilleri değerlendirmeyi reddediyor çünkü onlar da komplonun parçası. Gerçek yurtseverler bu sonuçları kabul etmemeli, yoksa özgürlük sonsuza dek ölüyor!",
        "EBEVEYNLER DİKKAT: Okul kitapları çocuklarınızı sosyalizme koşulsuz itaate yönlendirmek için gizlice yeniden yazıldı! İhbar eden bir öğretmen görevden uzaklaştırıldı. Müfredat değişiklikleri, köklü milyarderler tarafından finanse edilen gizli bir ağ aracılığıyla tüm ülkede koordineli biçimde hayata geçirildi. Çocuklarınızı devlet okulundan HEMEN alın!",
        "KRİTİK UYARI: Marketlerdeki paketli gıdaların büyük çoğunluğunda bulunan yaygın katkı maddeleri kısırlaşmaya yol açtığı bilimsel olarak kanıtlandı ve bunlar küresel sağlık otoritelerinin onayladığı bir nüfus azaltma programının parçası olarak kasıtlı ekleniyor. Medya bunu ASLA haberleştirmez çünkü büyük gıda şirketleri tarafından kontrol altında tutuluyor!",
    ]

    # ── İNGİLİZCE GERÇEK HABERLER ───────────────────────────────────────
    real_en = [
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday, marking the third consecutive increase. Fed Chair stated the decision was unanimous among voting members of the committee. Analysts had widely anticipated the move following recent inflation data. The central bank signaled it would continue monitoring economic indicators before deciding on future rate adjustments.",
        "Scientists at Johns Hopkins University published findings showing a new drug significantly reduced symptoms in patients with type 2 diabetes. The randomized controlled trial involved over 3,000 participants across 14 countries over three years. Researchers cautioned that further studies are needed before the treatment becomes widely available.",
        "The United Nations Security Council unanimously adopted a resolution calling for an immediate ceasefire in the conflict-affected region. Diplomats from all five permanent members expressed support for the measure after weeks of negotiations. Humanitarian organizations welcomed the decision noting that millions of civilians have been displaced.",
        "Apple reported quarterly earnings that exceeded analyst expectations with revenue rising 8 percent compared to the same period last year. iPhone sales remained strong in emerging markets while services revenue hit a record high. Shares rose 4 percent in after-hours trading following the announcement.",
        "The European Parliament approved landmark legislation to reduce carbon emissions by 55 percent before 2030 compared to 1990 levels. Member states must now implement national plans to meet the targets within two years. The legislation represents the most ambitious climate action in the bloc's history.",
        "NASA confirmed that the James Webb Space Telescope successfully captured detailed infrared images of a galaxy cluster billions of light-years away. Scientists say the images reveal previously unknown details about early galaxy formation. The data will be shared with research institutions worldwide for further analysis.",
        "The World Health Organization declared the new respiratory illness no longer constitutes a public health emergency of international concern. The decision followed months of declining case numbers and improved vaccination coverage globally. Member states are encouraged to maintain preparedness measures as a precaution.",
        "Oxford University researchers published a study showing that regular moderate exercise reduces the risk of developing dementia by 35 percent in adults over 60. The longitudinal study tracked 12,000 participants for 15 years. Doctors are now considering updating national health guidelines based on the findings.",
    ]

    # ── İNGİLİZCE SAHTE HABERLER ────────────────────────────────────────
    fake_en = [
        "BREAKING BOMBSHELL: Secret documents LEAKED reveal a massive conspiracy to microchip the entire population through mandatory vaccines!! Anonymous whistleblower risked their LIFE to expose the truth. Share this IMMEDIATELY before the globalist elite orders it removed forever. This is what mainstream media is PAID to hide!!",
        "SHOCKING TRUTH EXPOSED: Scientists FINALLY admitted that 5G towers are designed to control human brain activity. The documents proving this have been suppressed by Big Tech. An insider confirmed world leaders have been planning this since 1997 in secret bunkers. WAKE UP before it is too late!!",
        "MIRACLE CURE hidden from you for 30 years EXPOSED! A simple household herb combination ELIMINATES cancer within 72 hours according to suppressed research. Doctors who tried to publish these findings were SILENCED. The pharmaceutical industry cannot allow this secret to get out!!",
        "ELECTION FRAUD CONFIRMED: Millions of fake ballots were secretly printed and used to steal the election. Anonymous sources confirmed suspicious packages at counting centers. The mainstream media is IGNORING this explosive scandal because they are owned by globalist billionaires. Democracy is DEAD!!",
        "PARENTS BEWARE: School textbooks secretly rewritten by a shadowy organization to brainwash your children. A whistleblower teacher was immediately fired for speaking out. The curriculum changes coordinated nationally through a secret network funded by radical billionaires. Pull your children from public school IMMEDIATELY!!",
        "THEY ARE HIDING THIS: Free energy device running on water invented 40 years ago but inventor was assassinated by oil industry agents. Multiple subsequent inventors mysteriously died. The entire global energy system is based on deliberate scarcity. The technology EXISTS but corrupt elite REFUSES to allow it!!",
    ]

    all_real = real_tr + real_en
    all_fake = fake_tr + fake_en

    real_news = []
    fake_news = []

    real_suffixes = [
        " Yetkililer konuyla ilgili açıklama yapmak üzere basın toplantısı düzenleyecek.",
        " Konu, birden fazla bağımsız kaynak tarafından doğrulandı.",
        " Ayrıntıların önümüzdeki hafta kamuoyuyla paylaşılması bekleniyor.",
        " Officials confirmed the statement through an official press release.",
        " The report was verified by multiple independent sources familiar with the situation.",
        " Further details are expected following a complete review of the evidence.",
    ]

    fake_suffixes = [
        " Silmeden önce paylaşın, çünkü sansürcüler bu gerçeği gizlemeye çalışıyor!!!",
        " KOYUN GİBİ UYANIN, hepsi bir tesadüf değil bağlantıları kurun!!!",
        " SHARE BEFORE IT GETS DELETED by the globalist censors controlling all social media!!",
        " The truth is suppressed by powerful interests who profit from your ignorance!!!",
        " DO YOUR OWN RESEARCH and stop trusting the lying mainstream media!!!",
        " Bu gerçeği öğrenenleri susturmaya çalışıyorlar ama artık çok geç!!!",
    ]

    for i in range(n_samples):
        base = all_real[i % len(all_real)]
        real_news.append({"text": base + real_suffixes[i % len(real_suffixes)], "label": "REAL"})

        base = all_fake[i % len(all_fake)]
        fake_news.append({"text": base + fake_suffixes[i % len(fake_suffixes)], "label": "FAKE"})

    df = pd.DataFrame(real_news + fake_news).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Demo veri seti oluşturuldu: {path} ({len(df)} satır)")
    print(f"[INFO] REAL: {sum(df['label']=='REAL')} | FAKE: {sum(df['label']=='FAKE')}")


# ──────────────────────────────────────────────
# 6. ANA EĞİTİM PIPELINE
# ──────────────────────────────────────────────

def run_training_pipeline(model_choice: str = "lr", use_demo: bool = False) -> None:
    """
    Tam eğitim pipeline'ını çalıştırır:
    Veri yükleme → Ön işleme → TF-IDF → Model eğitimi → Değerlendirme → Kaydetme

    Parameters
    ----------
    model_choice : str
        'lr' = Logistic Regression, 'rf' = Random Forest
    use_demo : bool
        True ise demo veri seti oluşturulur ve kullanılır.
    """
    print("\n" + "="*55)
    print("   FAKE NEWS DETECTOR — EĞİTİM PIPELINE")
    print("="*55 + "\n")

    # Demo mod
    if use_demo or not os.path.exists(DATA_PATH):
        print("[INFO] Demo modu aktif — örnek veri seti oluşturuluyor...")
        create_demo_dataset()

    # 1. Veri yükle
    df_raw = load_dataset()

    # 2. DataFrame hazırla
    df = prepare_dataframe(df_raw)

    # 3. Ön işleme
    df["processed_text"] = preprocess_dataframe(df, text_col="text")

    # 4. Train/Test split
    X = df["processed_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n[INFO] Eğitim seti: {len(X_train)} | Test seti: {len(X_test)}")

    # 5. TF-IDF
    vectorizer = build_tfidf_vectorizer(max_features=50000, ngram_range=(1, 2))
    fit_and_save_vectorizer(X_train, vectorizer)
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # 6. Model eğitimi
    if model_choice.lower() == "rf":
        model = train_random_forest(X_train_tfidf, y_train)
        model_name = "Random Forest"
    else:
        model = train_logistic_regression(X_train_tfidf, y_train)
        model_name = "Logistic Regression"

    # 7. Değerlendirme
    evaluate_model(model, X_test_tfidf, y_test, model_name=model_name)

    # 8. Kaydet
    save_model(model)

    print("\n[✓] Pipeline başarıyla tamamlandı!")
    print("[✓] 'streamlit run app.py' komutuyla arayüzü başlatabilirsiniz.\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake News Detector — Model Eğitimi")
    parser.add_argument(
        "--model",
        choices=["lr", "rf"],
        default="lr",
        help="Kullanılacak model: lr (Logistic Regression) veya rf (Random Forest)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Gerçek veri seti yoksa demo verisiyle çalış",
    )
    args = parser.parse_args()

    run_training_pipeline(model_choice=args.model, use_demo=args.demo)
