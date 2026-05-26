# 🔍 Anomaly Detection with Autoencoders

Kredi kartı işlemlerinde dolandırıcılık tespiti için PyTorch tabanlı Autoencoder modeli.

## Proje Özeti

Bu proje, **denetimsiz öğrenme (unsupervised learning)** yaklaşımıyla anomali tespiti yapmaktadır.
Model yalnızca normal işlemlerle eğitilir; fraud işlemleri yüksek reconstruction error ürettiği
için anomali olarak işaretlenir.

## Sonuçlar

| Metrik | Değer |
|---|---|
| ROC-AUC | **0.9559** |
| F1 Score (Fraud) | **0.60** |
| Precision (Fraud) | 0.56 |
| Recall (Fraud) | 0.65 |
| Accuracy | 0.99 |

## Mimari
Girdi (29) → [16] → [8] → Bottleneck (4) → [8] → [16] → Çıktı (29)

## Kurulum

```bash
git clone https://github.com/barisaykent-spec/Anomaly_Detection.git
cd Anomaly_Detection
pip install -r requirements.txt
```

Veri setini [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)'dan indirip
`data/creditcard.csv` olarak kaydedin.

## Kullanım
notebooks/
├── 01_eda.ipynb          # Keşifsel veri analizi
├── 02_model.ipynb        # Model eğitimi
└── 03_evaluation.ipynb   # Metrikler ve grafikler
src/
├── model.py              # Autoencoder mimarisi
├── dataset.py            # Dataset ve DataLoader
└── utils.py              # Eğitim ve değerlendirme fonksiyonları

## Teknolojiler

- Python 3.x
- PyTorch (MPS — Apple M4)
- scikit-learn
- pandas / numpy / matplotlib / seaborn

## Veri Seti

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle

- 284.807 işlem
- %0.17 fraud oranı
- 30 feature (V1–V28 PCA, Amount, Time)

---

**Ders:** Yapay Zeka Uygulamaları | **Framework:** PyTorch