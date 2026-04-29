"""
create_turkish_datasets.py
--------------------------
LIAR Dataset ve Fake News Detection Dataset yapısını
taklit eden Türkçe veri setleri oluşturur.

LIAR Dataset  → Politik açıklamaları 6 sınıfa göre etiketler
                (pants-fire, false, barely-true, half-true, mostly-true, true)

Fake News Detection → Haber başlıkları + içerik (fake / real)

Her iki set de dataset.csv'ye eklenmek üzere REAL/FAKE formatına dönüştürülür.

Kullanım:
    python src/create_turkish_datasets.py
"""

import os
import pandas as pd
import numpy as np
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH  = os.path.join(DATA_DIR, "dataset.csv")
LIAR_PATH = os.path.join(DATA_DIR, "liar_turkish.csv")
FND_PATH  = os.path.join(DATA_DIR, "fnd_turkish.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LIAR DATASET — Türkçe Politik Açıklamalar
#     6 sınıf: pants-fire / false / barely-true / half-true / mostly-true / true
#     → FAKE: pants-fire, false, barely-true
#     → REAL: half-true, mostly-true, true
# ═══════════════════════════════════════════════════════════════════════════════

def create_liar_turkish(n_per_class: int = 120) -> pd.DataFrame:
    """
    LIAR veri setinin yapısını taklit eden Türkçe politik açıklama veri seti oluşturur.
    6 özgünlük seviyesini içerir; bunlar ikili FAKE/REAL etiketine dönüştürülür.

    Parameters
    ----------
    n_per_class : int
        Her 6 sınıf için üretilecek örnek sayısı.

    Returns
    -------
    pd.DataFrame
        text, liar_label, label sütunlarını içeren DataFrame.
    """

    # ── PANTS-FIRE (Tamamen Yalan) ──────────────────────────────────────────
    pants_fire = [
        "Türkiye dünyanın en yüksek vergi oranına sahip ülkesidir.",
        "Son seçimlerde oy kullanma oranı yüzde 20'nin altında kaldı.",
        "Emeklilik maaşları bu yıl yüzde 300 artırıldı.",
        "Türkiye hiçbir uluslararası iklim anlaşmasını imzalamamıştır.",
        "Benzin fiyatları son bir yılda yüzde 10 geriledi.",
        "Tüm devlet hastaneleri bu yıl özelleştirildi.",
        "Türkiye'de işsizlik oranı sıfıra indirildi.",
        "Asgari ücret son iki yılda üç katına çıktı.",
        "Bütün okullarda ücretsiz yemek uygulaması başladı.",
        "Ülkemizde son on yılda hiç vergi artışı yapılmadı.",
        "Tüm köylere fiber internet bağlantısı sağlandı.",
        "Kamu çalışanlarının tamamına ikinci maaş verildi.",
        "Son beş yılda sıfır trafik kazası ölümü kaydedildi.",
        "Türkiye en büyük yenilenebilir enerji üreticisi oldu.",
        "Tüm hastane masrafları devlet tarafından karşılanıyor.",
    ]

    # ── FALSE (Yanlış) ──────────────────────────────────────────────────────
    false_stmts = [
        "Muhalefet partisi geçen yıl iktidara geldi ve bütçeyi açığa taşıdı.",
        "Cumhurbaşkanı geçen ay vergiler yüzde 50 düşürüleceğini açıkladı.",
        "Yeni anayasa referandumu oy çokluğuyla reddedildi.",
        "Devlet, özel üniversitelerin tamamını kapattı.",
        "Son ekonomik rapor Türkiye'nin resesyona girdiğini gösteriyor.",
        "Bakanlık yabancı yatırımcılara yüzde 100 vergi muafiyeti tanıdı.",
        "Yeni yasa çıkarıldı, gece 10'dan sonra market alışverişi yasaklandı.",
        "Merkez Bankası faizleri sıfıra indirdi ve enflasyon geriledi.",
        "Ülkemizde suç oranları son on yılın en yüksek düzeyine ulaştı.",
        "Millet Bahçeleri projesi iptal edildi ve yerlerine AVM yapılacak.",
        "Devlet okullarında öğretmen maaşları yüzde 80 düşürüldü.",
        "Türk lirası bu yıl dolar karşısında yüzde 5 değer kazandı.",
        "Yeni yasa ile emeklilik yaşı 75'e yükseltildi.",
        "Devlet kanalları yıllık 10 milyar lira kâr ediyor.",
        "Sosyal yardım programları tamamen kaldırıldı.",
    ]

    # ── BARELY-TRUE (Çok Az Doğru) ──────────────────────────────────────────
    barely_true = [
        "Yatırımlarda ciddi artış yaşandı; ancak bu yalnızca belirli sektörlerde gerçekleşti.",
        "Hastane sayısı arttı, fakat personel eksiği ciddi sorunlara yol açmaya devam ediyor.",
        "Tarımsal ihracat rekor kırdı; bu rakamın büyük bölümü tek bir ürüne dayanıyor.",
        "Okullaşma oranı yükseldi, ancak eğitim kalitesi tartışmalı olmayı sürdürüyor.",
        "Suç istatistikleri düşüş gösterdi, ne var ki bazı kategorilerde artış gözlemlendi.",
        "Kamu yatırımları artış gösterdi; belirli bölgeler dışında yaygınlaşamadı.",
        "Yenilenebilir enerji kapasitesi genişledi, ithalata bağımlılık ise sürmekte.",
        "İstihdam rakamları iyileşti, gelir eşitsizliği sorunları henüz çözülmedi.",
        "Turizm gelirleri yükseldi, ancak dağılım bölgeden bölgeye büyük farklılık gösteriyor.",
        "Enflasyon bir miktar geriledi; temel gıda ürünlerindeki yükseliş ise sürmekte.",
        "Altyapı yatırımları arttı, kırsal kesimler bu gelişmeden yeterince yararlanamadı.",
        "Sağlık harcamaları yükseldi; fakat koruyucu sağlık hizmetleri yetersiz kalmaya devam ediyor.",
    ]

    # ── HALF-TRUE (Yarı Doğru) ──────────────────────────────────────────────
    half_true = [
        "İşsizlik oranı geçen yıla kıyasla düştü; ancak kayıt dışı istihdamı hesaba katmak gerekiyor.",
        "Eğitim bütçesi artırıldı; bununla birlikte enflasyona oranla reel artış sınırlı kaldı.",
        "Yeni hastaneler açıldı; yine de doktor başına düşen hasta sayısı hâlâ yüksek.",
        "Tarımsal destekler genişletildi, küçük çiftçilere ulaşımda güçlükler yaşanmaya devam etti.",
        "Altyapı harcamaları arttı; bazı projeler öngörülen takvimin gerisinde kaldı.",
        "Turizm rakamları rekor düzeye ulaştı; ancak gelirlerin dağılımı bölgeler arasında dengesiz.",
        "Yenilenebilir enerji yatırımları ilerledi, fosil yakıt kullanımı henüz tam olarak azalmadı.",
        "Sosyal yardım kapsamı genişledi, bununla beraber bazı hak sahipleri programlara erişemedi.",
        "İhracat rakamları yükseldi; bu artışın büyük payı yalnızca birkaç sektörden kaynaklanıyor.",
        "Eğitimde dijitalleşme hız kazandı; kırsal kesimlerde internet erişimi sınırlı kalmaya devam ediyor.",
    ]

    # ── MOSTLY-TRUE (Büyük Ölçüde Doğru) ──────────────────────────────────
    mostly_true = [
        "Türkiye son beş yılda yenilenebilir enerji kapasitesini önemli ölçüde artırdı.",
        "Sağlık sistemine yapılan yatırımlar artmış, ortalama yaşam beklentisi yükselmiştir.",
        "Eğitime ayrılan kaynaklar büyüdü, okullaşma oranlarında kayda değer ilerleme kaydedildi.",
        "Turizm gelirleri rekor seviyelere ulaştı ve istihdama önemli katkı sağladı.",
        "Altyapı projeleri büyük ölçüde tamamlandı; bazı gecikmeler yaşansa da çalışmalar sürmekte.",
        "Tarımsal ihracat artışı, çiftçi gelirlerinde gerçek bir iyileşmeye zemin hazırladı.",
        "Dijital devlet hizmetleri yaygınlaştı ve bürokrasinin azaltılmasına katkı sağladı.",
        "Özel sektör istihdamı artmış, ancak istihdam kalitesi bölgeden bölgeye farklılık göstermektedir.",
        "Kentsel dönüşüm projeleri konut kalitesini artırdı; yeniden yerleşim süreci zaman aldı.",
        "Savunma sanayiinde yerlilik oranı yükseldi ve önemli teknolojik atılımlar gerçekleştirildi.",
    ]

    # ── TRUE (Doğru) ────────────────────────────────────────────────────────
    true_stmts = [
        "Türkiye, NATO'nun en büyük ikinci kara kuvvetlerine sahip ülkesidir.",
        "İstanbul, Avrupa'nın en kalabalık şehirleri arasında yer almaktadır.",
        "Türkiye Büyük Millet Meclisi 600 milletvekili ile temsil edilmektedir.",
        "Türkiye, 2005 yılında Avrupa Birliği ile katılım müzakerelerini resmen başlattı.",
        "Türk lirası, 2021 yılında dolar karşısında ciddi bir değer kaybı yaşadı.",
        "Türkiye, dünyanın en fazla turist çeken ülkeleri arasında sürekli olarak yer almaktadır.",
        "Doğal afet riski taşıyan bir coğrafyada yer alan Türkiye, AFAD bünyesinde afet yönetim sistemleri kurmuştur.",
        "Türkiye, savunma teknolojileri alanında insansız hava araçları üretiminde önemli bir konuma gelmiştir.",
        "Yüksek öğretim alanında Türkiye'de devlet ve vakıf üniversitesi sayısı 200'ü aşmaktadır.",
        "Türkiye, sismik açıdan aktif bir bölgede bulunmakta olup büyük depremler yaşamaktadır.",
        "İstanbul Havalimanı, dünyanın en yoğun havalimanları arasına girmiştir.",
        "Türkiye, Ortadoğu ve Orta Asya ülkelerine yönelik insani yardım operasyonlarına ev sahipliği yapmaktadır.",
    ]

    rows = []

    classes = {
        "pants-fire":   (pants_fire,   "FAKE"),
        "false":        (false_stmts,  "FAKE"),
        "barely-true":  (barely_true,  "FAKE"),
        "half-true":    (half_true,    "REAL"),
        "mostly-true":  (mostly_true,  "REAL"),
        "true":         (true_stmts,   "REAL"),
    }

    for liar_label, (samples, binary_label) in classes.items():
        for i in range(n_per_class):
            base = samples[i % len(samples)]
            # Küçük varyasyonlar
            variations = [
                f" Bu açıklama {liar_label} kategorisinde değerlendirilmektedir.",
                f" Uzmanlar bu ifadeyi titizlikle inceledi.",
                f" Bağımsız kuruluşlar açıklamayı doğrulamak için araştırma yürüttü.",
                f" Kaynaklar bu iddianın arka planını mercek altına aldı.",
                f" Söz konusu açıklama kamuoyunda geniş yankı uyandırdı.",
            ]
            text = base + variations[i % len(variations)]
            rows.append({
                "text":        text,
                "liar_label":  liar_label,
                "label":       binary_label,
                "source":      "liar_turkish",
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[INFO] LIAR Türkçe: {len(df)} satır | FAKE: {sum(df['label']=='FAKE')} | REAL: {sum(df['label']=='REAL')}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FAKE NEWS DETECTION DATASET — Türkçe Haber Başlıkları + İçerik
# ═══════════════════════════════════════════════════════════════════════════════

def create_fnd_turkish(n_samples: int = 400) -> pd.DataFrame:
    """
    Fake News Detection Dataset yapısını taklit eden Türkçe
    haber başlığı + içerik çiftlerinden oluşan veri seti oluşturur.

    Parameters
    ----------
    n_samples : int
        Her sınıf (FAKE/REAL) için üretilecek örnek sayısı.

    Returns
    -------
    pd.DataFrame
        title, text, label, source sütunlarını içeren DataFrame.
    """

    # ── GERÇEK HABERLER: başlık + içerik ──────────────────────────────────
    real_pairs = [
        ("Merkez Bankası Faiz Oranını Sabit Tuttu",
         "Merkez Bankası Para Politikası Kurulu, beklentilerle uyumlu biçimde politika faizini yüzde 45'te sabit tuttu. Kurul açıklamasında enflasyondaki yavaşlama eğiliminin dikkatle izlendiği vurgulandı. Ekonomistler, mevcut verilerin faiz indirimini henüz desteklemediğini ifade etti. Bir sonraki toplantının tarihini açıkladı."),

        ("Türkiye Yenilenebilir Enerji Kapasitesini Artırdı",
         "Enerji Bakanlığı verilerine göre ülkenin rüzgâr ve güneş enerjisi kapasitesi geçen yıla kıyasla yüzde 18 artış kaydetti. Yeni güneş enerjisi santrallerinin devreye alınmasıyla birlikte toplam yenilenebilir enerji payı yüzde 42'ye yükseldi. Yetkililer bu oranın 2030 yılına kadar yüzde 60'a çıkarılmasını hedeflediklerini açıkladı."),

        ("İstanbul'da Metro Hattı Genişledi",
         "Büyükşehir belediyesi, yeni metro hattının açılışını gerçekleştirdi. Hat, 15 istasyonuyla günde 300 bin yolcu kapasitesine sahip. Proje, belirlenen bütçe dahilinde ve öngörülen sürede tamamlandı. Uzmanlar genişlemenin trafik yükünü önemli ölçüde hafifletmesini bekliyor."),

        ("Tarım İhracatı Rekor Düzeye Ulaştı",
         "Tarım Bakanlığı açıklamasına göre bu yılın ilk çeyreğinde tarımsal ihracat bir önceki yılın aynı dönemine kıyasla yüzde 24 artışla rekor kırdı. Büyüme; fındık, zeytinyağı ve taze sebze ihracatından kaynaklanıyor. Bakanlık, bu olumlu tablonun yıl boyunca sürmesini bekliyor."),

        ("Üniversite Araştırmacıları Kanser Tedavisinde Önemli Adım Attı",
         "Hacettepe Üniversitesi'nden araştırmacılar, bazı kanser türlerinde tümör büyümesini yavaşlatan yeni bir bileşik keşfetti. Hayvan deneyleri umut verici sonuçlar verirken insan denemeleri için başvurular yapıldı. Araştırma, uluslararası hakemli bir dergide yayımlandı ve bilim çevrelerinde geniş ilgi gördü."),

        ("Belediye Akıllı Şehir Projesini Hayata Geçiriyor",
         "Ankara Büyükşehir Belediyesi, yapay zeka tabanlı trafik yönetim sistemlerini pilot olarak uygulamaya başladı. İlk aşamada 50 kavşağa akıllı sensörler yerleştirildi. Veriler, trafik ışıklarını anlık koşullara göre optimize etmekte kullanılıyor. Proje, sürücü bekleme sürelerini yüzde 30 azaltmayı hedefliyor."),

        ("Doğal Afet Sigortası Kapsamı Genişletildi",
         "Hazine ve Maliye Bakanlığı, zorunlu deprem sigortası kapsamını genişleten yeni düzenlemeyi kamuoyuyla paylaştı. Değişiklikle ticari mülkler de kapsama alındı; prim hesaplamalarında güncel bina envanteri esas alınacak. Uzmanlar adımın olası büyük bir depremin ekonomik etkilerini önemli ölçüde azaltacağını belirtiyor."),

        ("Türk Turizmi Yeni Rekora İmza Attı",
         "Kültür ve Turizm Bakanlığı verilerine göre bu yıl yabancı ziyaretçi sayısı 55 milyonu aştı. Gelirlerin ise 60 milyar dolara yaklaştığı açıklandı. Antalya, İstanbul ve Kapadokya en çok tercih edilen destinasyonlar oldu. Sektör temsilcileri, 2024 ve sonrası için iyimser bir tablo çiziyor."),

        ("Eğitimde Dijital Dönüşüm Hız Kazandı",
         "Millî Eğitim Bakanlığı, EBA platformunun 15 milyonu aşkın aktif kullanıcıya ulaştığını açıkladı. Dijital içerik kütüphanesi 50 binden fazla eğitim materyalini bünyesinde barındırıyor. Bakanlık, tüm sınıflara akıllı tahta yerleştirme çalışmalarını sürdürdüğünü bildirdi. Öğretmenler için çevrimiçi mesleki gelişim programları da devam ediyor."),

        ("Savunma Sanayii Yerlilik Oranı Yüzde 80'i Aştı",
         "Savunma sanayii yöneticileri, savunma tedarikindeki yerlilik oranının yüzde 80'i geçtiğini duyurdu. Bu oran, on yıl öncesinin yüzde 20 seviyesinden büyük bir sıçramayı temsil ediyor. İnsansız hava araçları ve zırhlı araçlar bu başarının öncü alanları arasında yer aldı. Sektör, ihracat rakamlarını katlayarak büyütmeyi hedefliyor."),

        ("Sağlıkta Dijital Dönüşüm Hasta Memnuniyetini Artırdı",
         "Sağlık Bakanlığı, elektronik reçete ve e-Nabız sisteminin kullanımının yüzde 90'a ulaştığını açıkladı. Dijital randevu sistemleri, hastalarda bekleme sürelerini belirgin biçimde kısalttı. Anketler, dijital hizmetlere geçişin ardından hasta memnuniyetinin yüzde 35 arttığını ortaya koyuyor. Bakanlık, yapay zeka destekli tanı sistemlerini de devreye almayı planlıyor."),

        ("İstanbul Finans Merkezi Uluslararası Yatırımcıları Çekiyor",
         "İstanbul Finans Merkezi, ilk yılında 50'yi aşkın uluslararası finans kuruluşunu bünyesine kattı. Bölge, ülkenin finansal merkez olma hedefi doğrultusunda önemli bir adım olarak değerlendiriliyor. Merkez yönetimi 500 milyon dolarlık yabancı yatırım çekmeyi başardığını açıkladı. İstihdam rakamları da beklentileri aştı."),
    ]

    # ── SAHTE HABERLER: başlık + içerik ──────────────────────────────────
    fake_pairs = [
        ("ŞOK İDDİA: Hükümet Vatandaşları İzliyor, Telefon Kameraları 7/24 Aktif!",
         "Anonim bir yazılım mühendisi, devletin tüm akıllı telefonların kameralarına uzaktan erişebildiğini iddia ediyor. Bu kişinin paylaştığı belgelerin gerçekliği tartışmalı; bağımsız uzmanlar doğrulayamadı. Ana akım medya haberi görmezden geliyor çünkü hükümetle iş birliği yapıyorlar. Bu haberi silmeden önce acilen paylaşın!"),

        ("BOMBA: İçme Suyuna Kitlesel Uyuşturucu Madde Karıştırıldığı İddia Edildi",
         "Güvenilmez bir kaynaktan edinilen iddialara göre büyükşehirlerdeki su şebekelerine nüfus kontrolü amacıyla kimyasal madde eklenmiş. Uzmanlar bu iddiayı kesinlikle reddetti ve söz konusu kanalların standart denetimden geçtiğini vurguladı. Sağlık Bakanlığı yetkilileri iddiaları tamamen asılsız bulduklarını açıkladı. Viral olan paylaşımlar hiçbir kaynağa dayanmıyor."),

        ("GİZLİ: Aşıların İçinde Takip Çipi Bulunuyor, Doktorlar SustU!",
         "Sosyal medyada hızla yayılan iddialara göre COVID-19 aşıları insan vücuduna nano boyutlu takip cihazı enjekte ediyor. Tıp uzmanları, biyologlar ve mühendisler bu senaryonun mevcut teknoloji kapsamında imkânsız olduğunu defalarca açıkladı. Dünya Sağlık Örgütü ve Sağlık Bakanlığı doğrulama mekanizmalarıyla aşıların güvenli olduğunu teyit etti. Dezenformasyon nedeniyle aşılanma oranlarının düştüğü bölgelerde salgın yeniden alevleniyor."),

        ("PATLAMA: Ünlü İş İnsanı Hükümete Para Akıttı, İşte Belgeler!",
         "Kimliği bilinmeyen bir kaynak, tanınmış bir iş insanının siyasi partilere yasadışı bağış yaptığını öne sürdü. Belgeler manipüle edilmiş görünüyor ve gerçekliği bağımsız kuruluşlar tarafından doğrulanamadı. Söz konusu iş insanının avukatları tüm iddiaları kesinlikle reddetti. Haberi asıl yayanlara ulaşılmaya çalışıldı ancak kaynaklara erişilemedi."),

        ("DIKKAT: 5G Kuleleri Kanser Yapıyor, Bilim İnsanları Sustuk!",
         "Sosyal medyada viral olan paylaşımlar, 5G kulelerinin yüksek frekanslı radyasyon yayarak kansere neden olduğunu ileri sürüyor. Dünya Sağlık Örgütü ve uluslararası ışınım kuruluşları, radyasyonun güvenli sınırların çok altında kaldığını defalarca açıkladı. Bu teknolojinin sağlığa etkisi onlarca yıldır kapsamlı biçimde araştırılıyor. Uzmanlar, korkuların bilimsel dayanaktan yoksun olduğunu vurguluyor."),

        ("GİZLİ GÜNDEM: Büyük Teknoloji Firmaları Düşüncelerinizi Okuyor!",
         "Hiçbir kaynağa dayandırılmayan iddialara göre Apple, Google ve Meta çeşitli gizli algoritmalar aracılığıyla kullanıcıların düşüncelerini okuyabiliyor. Siber güvenlik uzmanları bu tür iddiaların mevcut teknoloji ile uyumsuz olduğunu belirtti. Söz konusu şirketler iddiaları kesinlikle reddetti. Yine de viral paylaşımlar internette yayılmaya devam ediyor."),

        ("SKANDAL: Ünlüler Gizli Cemiyetin Üyesi, İşte Kanıtlar!",
         "Anonim bir kaynaktan yayılan iddialara göre Türkiye'nin tanınmış isimleri, toplumu şekillendirdiği öne sürülen gizli bir örgütün üyesi. Haber doğrulama platformları iddiaların asılsız olduğunu tespit etti. Paylaşılan fotoğraf ve videolar bağlamından koparılarak ya da dijital olarak düzenlenmiş. İlgili kişilerin avukatları soruşturma başlatıldığını duyurdu."),

        ("ACİL UYARI: Marketlerdeki Bu Ürün Sizi Zehirliyor!",
         "Sosyal medyada hızla yayılan bir videoda tanınmış bir marka ürününün tüketilmesi halinde ciddi sağlık sorunlarına yol açacağı ileri sürülüyor. Gıda güvenliği uzmanları iddiaları inceleyerek asılsız buldu. İlgili ürün tüm yasal standartları karşılıyor; düzenli denetime tabi tutuluyor. Marka, yasal süreç başlatacağını açıkladı."),

        ("BOMBA: Büyük Banka İflas Eşiğinde, Paranızı Çekin!",
         "Sosyal medyada belirli bir bankanın iflasın eşiğine geldiğine ilişkin haberler yayılıyor. Bankacılık Düzenleme ve Denetleme Kurumu bankacılık sisteminin güçlü ve sağlıklı olduğunu açıkladı. İlgili bankanın mali göstergeleri son derece olumlu seyrediyor. Söz konusu dezenformasyon, banka operasyonlarının istikrarını tehdit etmektedir."),

        ("GİZLİ: NASA Dünya'ya Yaklaşan Asteroiti Saklıyor!",
         "İnternette dolaşan iddialara göre NASA büyük bir asteroitin Dünya'ya çarpacağını bilerek kamuoyundan gizliyor. NASA ve diğer uzay ajansları gezegen savunma programlarını kamuoyuyla şeffaf biçimde paylaşıyor. Bilim insanları, bu tür iddiaların zaten mevcut olan açık verilerle çeliştiğini vurguluyor. Yine de komplo teorileri sosyal medyada yayılmaya devam ediyor."),
    ]

    rows = []

    for i in range(n_samples):
        title, content = real_pairs[i % len(real_pairs)]
        text = title + " " + content
        suffix_list = [
            " Konu yetkili makamlar tarafından teyit edildi.",
            " Uzmanlar değerlendirmelerini basın toplantısıyla kamuoyuyla paylaştı.",
            " Resmi açıklama bakanlık web sitesinde yayımlandı.",
            " Bağımsız doğrulama kuruluşları haberin doğru olduğunu onayladı.",
        ]
        rows.append({
            "title":  title,
            "text":   text + suffix_list[i % len(suffix_list)],
            "label":  "REAL",
            "source": "fnd_turkish",
        })

    for i in range(n_samples):
        title, content = fake_pairs[i % len(fake_pairs)]
        text = title + " " + content
        suffix_list = [
            " Silinmeden önce acilen paylaşın!!!",
            " Medya bunu görmezden geliyor, siz yayın!!!",
            " Gerçeği gizlemek istiyorlar, sessiz kalmayın!!!",
            " Bu haberi gören herkes şaşkına dönüyor!!!",
        ]
        rows.append({
            "title":  title,
            "text":   text + suffix_list[i % len(suffix_list)],
            "label":  "FAKE",
            "source": "fnd_turkish",
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[INFO] FND Türkçe: {len(df)} satır | FAKE: {sum(df['label']=='FAKE')} | REAL: {sum(df['label']=='REAL')}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  VERİLERİ BİRLEŞTİR VE KAYDET
# ═══════════════════════════════════════════════════════════════════════════════

def merge_and_save(save_individual: bool = True) -> None:
    """
    LIAR ve FND Türkçe veri setlerini oluşturur; varsa mevcut
    dataset.csv ile birleştirerek kaydeder.

    Parameters
    ----------
    save_individual : bool
        True ise her iki veri setini ayrı CSV olarak da kaydeder.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "="*55)
    print("  TÜRKÇE VERİ SETLERİ OLUŞTURULUYOR")
    print("="*55 + "\n")

    # 1. Veri setlerini oluştur
    df_liar = create_liar_turkish(n_per_class=120)
    df_fnd  = create_fnd_turkish(n_samples=400)

    # 2. İsteğe bağlı: ayrı ayrı kaydet
    if save_individual:
        df_liar[["text", "label"]].to_csv(LIAR_PATH, index=False)
        print(f"[INFO] Kaydedildi: {LIAR_PATH}")

        df_fnd[["text", "label"]].to_csv(FND_PATH, index=False)
        print(f"[INFO] Kaydedildi: {FND_PATH}")

    # 3. Yalnızca text + label sütunlarını al
    new_data = pd.concat([
        df_liar[["text", "label"]],
        df_fnd[["text",  "label"]],
    ], ignore_index=True)

    # 4. Mevcut dataset.csv varsa birleştir
    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        # Metin sütununu bul
        text_col = next((c for c in existing.columns if c.lower() in ["text","content","article"]), None)
        label_col = next((c for c in existing.columns if c.lower() == "label"), None)

        if text_col and label_col:
            existing_clean = existing[[text_col, label_col]].copy()
            existing_clean.columns = ["text", "label"]
            combined = pd.concat([existing_clean, new_data], ignore_index=True)
        else:
            combined = new_data
            print("[WARN] Mevcut dataset.csv uyumlu sütun içermiyor, sadece yeni veri kullanılıyor.")
    else:
        combined = new_data
        print("[INFO] Mevcut dataset.csv bulunamadı, yalnızca Türkçe veri kullanılıyor.")

    # 5. Karıştır ve kaydet
    combined = combined.dropna().sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"  BİRLEŞTİRME TAMAMLANDI")
    print(f"{'='*55}")
    print(f"  Toplam satır : {len(combined)}")
    print(f"  REAL         : {sum(combined['label']=='REAL')}")
    print(f"  FAKE         : {sum(combined['label']=='FAKE')}")
    print(f"  Kaydedildi   : {OUT_PATH}")
    print(f"\n[✓] Şimdi modeli yeniden eğitebilirsiniz:")
    print(f"    python src/train_model.py --model lr\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    merge_and_save(save_individual=True)
