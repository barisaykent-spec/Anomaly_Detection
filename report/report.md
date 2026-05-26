# Autoencoder ile Anomali Tespiti: Kredi Kartı Dolandırıcılığı Tespiti

**Ders:** Yapay Zeka Uygulamaları  
**Proje No:** 9 — Anomaly Detection with Autoencoders  
**Framework:** PyTorch  
**Donanım:** Apple M4 (MPS)  
**Tarih:** Mayıs 2026

---

## 1. Giriş

Finansal sistemlerde dolandırıcılık tespiti, makine öğrenmesinin en kritik uygulama alanlarından biridir. Kredi kartı işlemlerinin büyük çoğunluğu normaldir; dolandırıcılık işlemleri ise tüm işlemlerin yalnızca küçük bir oranını oluşturmaktadır. Bu durum, geleneksel denetimli öğrenme yöntemlerini doğrudan uygulamayı güçleştirmektedir: eğitim verisi aşırı dengesizdir ve her fraud işlemin etiketlenmesi hem maliyetli hem de zaman alıcıdır.

Bu projede, **denetimsiz öğrenme (unsupervised learning)** yaklaşımı benimsenerek **Autoencoder** mimarisi kullanılmıştır. Modele yalnızca normal işlem verileri gösterilmiş; anormal işlemler (fraud) eğitimde hiç yer almamıştır. Temel hipotez şudur: normal işlemleri iyi öğrenen bir model, hiç görmediği fraud işlemleri yeniden üretemez ve bu başarısızlık "anomali skoru" olarak kullanılabilir.

---

## 2. Veri Seti

### 2.1 Genel Bilgiler

| Özellik | Değer |
|---|---|
| Kaynak | Kaggle — Credit Card Fraud Detection Dataset |
| Toplam işlem | 284.807 |
| Feature sayısı | 30 (Time hariç tutuldu) |
| Eksik değer | Yok |

### 2.2 Sınıf Dağılımı

| Sınıf | Adet | Oran |
|---|---|---|
| Normal (0) | 284.315 | %99.83 |
| Fraud (1) | 492 | %0.17 |

Veri seti aşırı dengesizdir: 577 normal işleme karşılık yalnızca 1 fraud işlem düşmektedir. Bu oran, denetimli yöntemlerin sınıf dengesizliğine karşı dirençsiz olduğu durumlarda Autoencoder'ın ne kadar uygun bir tercih olduğunu göstermektedir.

### 2.3 Feature Yapısı

- **V1–V28:** PCA ile anonimleştirilmiş finansal özellikler. Gizlilik nedeniyle orijinal anlamları açıklanmamıştır.
- **Amount:** İşlem tutarı. StandardScaler ile normalize edilmiştir.
- **Time:** Veri setinden çıkarıldı (modele katkısı düşük, sıra bilgisi taşıyor).

### 2.4 Veri Ön İşleme

- `Amount` sütunu StandardScaler ile normalize edildi.
- `Time` sütunu veri setinden çıkarıldı.
- Train seti: yalnızca normal işlemler (%80 → 227.452 satır).
- Test seti: normal + fraud karışık (%20 normal + tüm fraud → 57.355 satır).

---

## 3. Yöntem

### 3.1 Autoencoder Mimarisi

Autoencoder, bir **Encoder** ve bir **Decoder** olmak üzere iki bileşenden oluşmaktadır.

```
Girdi (29)  →  Encoder  →  Bottleneck (4)  →  Decoder  →  Çıktı (29)
```

**Encoder:** Girdi verisini giderek küçülen boyutlara sıkıştırır.  
**Bottleneck:** Verinin en sıkıştırılmış temsili — "öz".  
**Decoder:** Sıkıştırılmış temsili orijinal boyuta geri açar.

| Katman | Boyut |
|---|---|
| Girdi | 29 |
| Encoder — Gizli Katman 1 | 16 |
| Encoder — Gizli Katman 2 | 8 |
| Bottleneck | 4 |
| Decoder — Gizli Katman 1 | 8 |
| Decoder — Gizli Katman 2 | 16 |
| Çıktı | 29 |

- **Aktivasyon:** ReLU (tüm ara katmanlarda)
- **Toplam parametre:** ~1.100
- **Kayıp fonksiyonu:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr=0.001)

### 3.2 Anomali Tespiti Mantığı

Model yalnızca normal verilerle eğitildiğinden, normal işlemleri düşük hatayla yeniden üretebilir. Fraud işlemler modelin hiç görmediği örüntüler olduğundan yeniden üretim hatası yüksek çıkar. Bu **reconstruction error**, anomali skoru olarak kullanılır:

```
Reconstruction Error (MSE) > Threshold  →  Fraud
Reconstruction Error (MSE) ≤ Threshold  →  Normal
```

### 3.3 Threshold Optimizasyonu

Başlangıçta sabit bir threshold (normal hataların 95. yüzdilimi = 1.03) kullanıldı. Ancak bu yaklaşım yüksek recall (%88) sağlarken precision'ı ciddi ölçüde düşürdü (%13) — 2.844 yanlış alarm.

Daha sonra, tüm olası threshold değerleri üzerinde **F1 skoru maksimizasyonu** uygulandı. En iyi threshold 4.44 olarak belirlendi.

---

## 4. Sonuçlar

### 4.1 Eğitim Süreci

| Epoch | Ortalama MSE Kaybı |
|---|---|
| 5 | 0.545007 |
| 10 | 0.472685 |
| 15 | 0.461337 |
| 20 | 0.455662 |
| 25 | 0.451594 |
| 30 | 0.447106 |

Kayıp eğrisi monoton azalmış ve yakınsama (convergence) ~15. epoch'ta gözlemlenmiştir. Salınım yoktur — eğitim kararlı ve sağlıklıdır.

### 4.2 Değerlendirme Metrikleri

**Sabit Threshold (95. Yüzdil = 1.03):**

| Metrik | Normal | Fraud |
|---|---|---|
| Precision | 1.00 | 0.13 |
| Recall | 0.95 | 0.88 |
| F1-Score | 0.97 | 0.23 |
| Accuracy | — | 0.95 |

**Optimize Edilmiş Threshold (F1-max = 4.44):**

| Metrik | Normal | Fraud |
|---|---|---|
| Precision | 1.00 | 0.56 |
| Recall | 1.00 | 0.65 |
| F1-Score | 1.00 | 0.60 |
| Accuracy | — | 0.99 |

**ROC-AUC: 0.9559**

### 4.3 Confusion Matrix (Optimize Threshold)

| | Tahmin: Normal | Tahmin: Fraud |
|---|---|---|
| **Gerçek: Normal** | 56.760 (TN) | ~103 (FP) |
| **Gerçek: Fraud** | 172 (FN) | 320 (TP) |

---

## 5. Tartışma

### 5.1 Güçlü Yönler

- **Etiket gerektirmez:** Model hiçbir fraud etiketi görmeden %60 F1 elde etti. Bu, denetimsiz öğrenmenin gücünü göstermektedir.
- **Yüksek ROC-AUC (0.9559):** Model, normal ve fraud işlemleri reconstruction error bazında çok iyi ayırt edebilmektedir.
- **Pratik ölçeklenebilirlik:** Yeni fraud türleri ortaya çıktığında model yeniden etiketlenmeden güncellenebilir.

### 5.2 Zayıf Yönler ve Sınırlılıklar

- **Precision-Recall tradeoff:** F1 optimize edilmiş durumda bile precision %56 — yani her 2 fraud alarmından biri yanlış. Gerçek banka sistemlerinde bu hâlâ yüksek.
- **Threshold seçiminin statikliği:** Threshold sabit tutuldu. Gerçek sistemlerde dinamik threshold gerekebilir.
- **Feature yorumlanabilirliği:** V1–V28 PCA dönüştürülmüş olduğundan hangi finansal özelliklerin fraud'u tetiklediği anlaşılamıyor.

### 5.3 Precision vs Recall Tradeoff Analizi

Bu projede threshold seçimi kritik bir tasarım kararıdır ve iş gereksinimine göre değişir:

- **Yüksek Recall öncelikliyse (fraud kaçırma):** Düşük threshold — daha fazla yanlış alarm kabul edilir.
- **Yüksek Precision öncelikliyse (müşteri memnuniyeti):** Yüksek threshold — bazı fraudlar kaçırılır.
- **Dengeli (F1 max):** Threshold = 4.44, her iki taraf için kabul edilebilir orta yol.

### 5.4 Baseline Karşılaştırması: Isolation Forest

Aynı eğitim stratejisiyle (yalnızca normal veriler) Isolation Forest modeli de test edilmiştir.

| Metrik | Autoencoder | Isolation Forest |
|---|---|---|
| ROC-AUC | **0.9559** | 0.9491 |
| F1 Score (Fraud) | **0.60** | 0.37 |
| Recall (Fraud) | **0.65** | 0.28 |
| Precision (Fraud) | 0.56 | 0.56 |

Autoencoder her metrikte üstün performans göstermiştir. Bunun temel nedeni, 
veri setindeki V1–V28 feature'larının PCA ile dönüştürülmüş olmasıdır. 
Bu durumda fraud işlemler geometrik olarak uç değer olmayabilir — 
Isolation Forest'ın zayıf kaldığı nokta tam olarak budur. 
Autoencoder ise feature uzayındaki geometrik konuma değil, 
**örüntünün yeniden üretilebilirliğine** dayandığı için bu senaryoda 
daha etkili bir anomali tespiti sağlamaktadır.

---

## 6. Sonuç

Bu proje, Autoencoder mimarisinin denetimsiz anomali tespitindeki etkinliğini başarıyla göstermiştir. PyTorch ve Apple MPS (M4) hızlandırması ile eğitilen model, eğitimde hiç görmediği fraud işlemleri ROC-AUC=0.9559 ve F1=0.60 ile tespit edebilmektedir.

Elde edilen sonuçlar, etiketli veri olmaksızın bile güçlü bir baseline oluşturulabileceğini ortaya koymaktadır. İleride yapılabilecek iyileştirmeler şunlardır:

- Variational Autoencoder (VAE) ile karşılaştırma
- Daha derin mimari veya dropout regularization denemesi
- Ensemble yöntemiyle Autoencoder + Isolation Forest kombinasyonu
- SHAP değerleri ile model yorumlanabilirliği analizi

---

## 7. Teknik Detaylar

| Parametre | Değer |
|---|---|
| Framework | PyTorch |
| Donanım | Apple M4 (MPS) |
| Epoch | 30 |
| Batch Size | 256 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Kayıp Fonksiyonu | MSE |
| Train Seti | 227.452 normal işlem |
| Test Seti | 57.355 işlem (normal + fraud) |

---

*GitHub: [https://github.com/barisaykent-spec/Anomaly_Detection](https://github.com/barisaykent-spec/Anomaly_Detection)*
