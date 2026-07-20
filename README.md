# Semptom Verilerine Dayalı Hastalık Tahmin ve Karar Destek Sistemi

Bu proje, semptom verilerinden hastalık tahmini yapan modüler bir karar destek sistemidir. Sistem arka planda `Apriori`, `Decision Tree`, `Naive Bayes` ve `Random Forest` modellerini birlikte çalıştırır ve sonuçları tek bir Streamlit arayüzünde gösterir.

## Özellikler

- Semptom bazlı seçim ekranı
- Apriori ile semptomdan hastalığa giden kurallar
- Decision Tree, Naive Bayes ve Random Forest tahminleri
- Naive Bayes olasılık grafiği
- Random Forest feature importance grafiği
- Hastalık dağılımı ve en sık görülen semptom grafikleri
- Joblib ile model kaydı

## Proje Yapısı

```text
DiseasePrediction/
├── data/
│   ├── Training.csv
│   ├── Testing.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
├── models/
├── src/
│   ├── preprocess.py
│   ├── apriori_rules.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── predict.py
│   ├── visualization.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## Kurulum

1. Python 3.12+ kurun.
2. Sanal ortam oluşturun ve etkinleştirin.
3. Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

## Veri Dosyaları

Kod, veri dosyalarını önce `data/` klasöründe arar. Dosyalar doğrudan proje kökünde bulunuyorsa onları da otomatik olarak bulur.

Önerilen konum:

- `data/Training.csv`
- `data/Testing.csv`
- `data/symptom_Description.csv`
- `data/symptom_precaution.csv`
- `data/Symptom-severity.csv`

## Uygulamayı Çalıştırma

```bash
streamlit run app.py
```

İlk çalıştırmada modeller bulunmazsa sistem `Training.csv` üzerinden otomatik eğitim yapar ve modeli `models/` klasörüne kaydeder.

## Çıktılar

- `decision_tree.pkl`
- `naive_bayes.pkl`
- `random_forest.pkl`

## Not

Bu sistem yalnızca karar destek amaçlıdır. Kesin tanı yerine geçmez.