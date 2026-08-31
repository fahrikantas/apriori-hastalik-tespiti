#  Semptom Verilerine Dayalı Hastalık Tahmin ve Karar Destek Sistemi

Semptom verilerinden olası hastalıkları tahmin etmek amacıyla geliştirilen, **makine öğrenmesi, birliktelik analizi, topluluk öğrenmesi ve açıklanabilir yapay zekâ** yöntemlerini bir araya getiren modüler bir karar destek sistemidir.

Sistem; kullanıcının seçtiği semptomları analiz ederek birden fazla makine öğrenmesi modelinden tahminler üretir, modellerin sonuçlarını **ağırlıklı soft-voting** yöntemiyle birleştirir ve sonuçları güven seviyesi, ICD-10 kodu, ayırıcı tanılar, Apriori kuralları ve açıklanabilirlik çıktılarıyla birlikte sunar.

>  **Önemli:** Bu proje tıbbi karar destek amacıyla geliştirilmiştir ve profesyonel tıbbi tanının yerine geçmez. Gerçek sağlık sorunlarında mutlaka bir sağlık profesyoneline başvurulmalıdır.

---

## Projenin Amacı

Klasik hastalık tahmin sistemlerinde tek bir makine öğrenmesi modelinin kullanılması, modelin güçlü ve zayıf yönlerine bağlı olarak tahmin performansını sınırlayabilir.

Bu projede ise farklı algoritmaların tahminleri bir araya getirilerek daha kapsamlı bir karar destek mekanizması oluşturulmuştur.

Sistem temel olarak şu süreci takip eder:

```text
Kullanıcı Semptomları
        │
        ▼
Semptom Ön İşleme
        │
        ├──────────────► Apriori Kuralları
        │
        ▼
┌───────────────────────────────┐
│       Makine Öğrenmesi        │
│                               │
│ Decision Tree                 │
│ Naive Bayes                   │
│ Random Forest                 │
│ Logistic Regression           │
│ SVM                           │
│ XGBoost                       │
│ LightGBM                      │
└───────────────────────────────┘
        │
        ▼
 Ağırlıklı Soft-Voting
        │
        ▼
 Nihai Hastalık Tahmini
        │
        ├──► Güven Seviyesi
        ├──► ICD-10 Kodu
        ├──► Ayırıcı Tanılar
        ├──► OOD Kontrolü
        ├──► Red Flag Kontrolü
        ├──► SHAP / LIME
        └──► Raporlama
```

---

# Özellikler

## Çoklu Makine Öğrenmesi

Sistem aynı semptom girdisini **7 farklı sınıflandırma modeli** ile analiz eder:

* Decision Tree
* Naive Bayes
* Random Forest
* Logistic Regression
* Support Vector Machine (SVM)
* XGBoost
* LightGBM

Modellerin tahminleri daha sonra topluluk öğrenmesi yaklaşımıyla birleştirilir.

---

## Apriori Birliktelik Kuralları

Makine öğrenmesi modellerinin yanında **Apriori algoritması** kullanılarak semptom-hastalık ilişkileri analiz edilir.

Örneğin:

```text
Semptom A + Semptom B
          │
          ▼
     Hastalık X
```

Apriori sonuçları, seçilen semptomların hangi hastalıklarla daha güçlü biçimde ilişkili olduğunu göstermeye yardımcı olur.

---


## Ağırlıklı Soft-Voting Ensemble

Her modelin tahmini doğrudan eşit ağırlıkta kullanılmaz.

Model performanslarına göre ağırlıklandırılmış bir **soft-voting ensemble** mekanizması kullanılır.

Bu yapı sonucunda:

* Nihai hastalık tahmini
* Hastalık olasılıkları
* Model bazlı tahminler
* Modeller arasındaki görüş ayrılığı

birlikte değerlendirilebilir.

`predict_proba` desteklemeyen modeller için sistem uygun bir fallback mekanizması kullanır.

---

## Adaptif Belirti Toplama

Sistem yalnızca kullanıcının verdiği semptomlarla sınırlı kalmaz.

**"Sıradaki belirtiyi sor"** mekanizması ile henüz seçilmemiş semptomlar arasından bilgi kazancı yüksek olan semptom önerilebilir.

Bu mekanizma:

* Mutual Information
* Mevcut semptomlar
* Önceki tahminler

gibi bilgilerden yararlanarak daha odaklı semptom toplama süreci oluşturur.

---

## Hastalık Tahmini ve Ayırıcı Tanı

Sistem yalnızca tek bir hastalık döndürmek yerine birden fazla olası hastalığı değerlendirebilir.

Çıktılarda:

* Nihai tahmin
* En olası hastalıklar
* Top-5 ayırıcı tanı
* Her tanıyı destekleyen modeller
* Model skorları

gösterilebilir.

---

## Güven Seviyesi

Tahmin sonucuna göre güven seviyesi oluşturulur:

| Seviye    | Açıklama     |
| --------- | ------------ |
| 🟢 High   | Yüksek güven |
| 🟡 Medium | Orta güven   |
| 🔴 Low    | Düşük güven  |

Güven seviyesi özellikle modellerin birbirinden farklı sonuçlar ürettiği durumlarda kullanıcıya ek bilgi sağlar.

---


## OOD (Out-of-Distribution) Algılama

Girilen semptom kombinasyonunun eğitim verilerine ne kadar benzediği kontrol edilir.

Sistem, eğitim verisinden oldukça farklı bir semptom kombinasyonu tespit ettiğinde kullanıcıya **OOD uyarısı** verebilir.

Bu özellik, modelin eğitim verisinin dışında kalan örneklerde aşırı güvenli sonuçlar üretmesini önlemeye yardımcı olur.

---

##Red Flag (Kritik Belirti) Kontrolü

Sistemde hastalık tahmininden bağımsız olarak çalışan kural tabanlı bir Red Flag güvenlik mekanizması bulunmaktadır.Sistem belirli kritik semptomları ve semptom kombinasyonlarını kontrol eder.

Bu mekanizmanın amacı, makine öğrenmesi modeli herhangi bir hastalığı düşük olasılıkla tahmin etse bile, kullanıcının girdiği semptomlar arasında acil tıbbi değerlendirme gerektirebilecek kritik belirtileri ayrıca kontrol etmektir.

Bazı belirtiler tek başına kritik olmayabilir; ancak belirli semptomların birlikte görülmesi daha ciddi bir durumun göstergesi olabilir.
Bu nedenle sistem semptomları yalnızca tek tek değil, kombinasyon halinde de kontrol eder.
Örneğin:

```text
Yüksek ateş + Boyun tutulması
              ↓
       Kritik durum uyarısı
```

veya:

```text
Tek taraflı güçsüzlük + Konuşma bozukluğu
              ↓
       Acil değerlendirme uyarısı
```

Red flag kuralları:

* Web arayüzünde
* REST API'de
* Sohbet asistanında

ortak şekilde kullanılmaktadır.

---

## ICD-10 Entegrasyonu

Tahmin edilen hastalıklar **ICD-10** kodlarıyla eşleştirilir.

ICD-10 kodları:

* Tahmin sonuçlarında
* Ayırıcı tanılarda
* Raporlarda

gösterilebilir.

Projede hastalık etiketleri için kapsamlı ICD-10 eşlemesi bulunmaktadır.

---

## Türkçe Sohbet Asistanı

Projede semptom ve hastalık bilgileriyle etkileşim kurulabilmesini sağlayan Türkçe sohbet asistanı bulunmaktadır.

Sistem:

* Serbest metin semptomlarını algılayabilir
* Türkçe semptom adlarını destekler
* Semptom/hastalık bilgileri sunabilir
* Konuşma bağlamını takip edebilir
* Kullanıcı geri bildirimi alabilir

---

## Türkçe / İngilizce Dil Desteği

Arayüz iki dili destekler:

* 🇹🇷 Türkçe
* 🇬🇧 English

Çeviri metinleri `src/i18n.py` içerisinde merkezi olarak yönetilir.

---

##Semptom Şiddeti ve Süresi

Kullanıcı seçtiği semptomlar için:

* Şiddet
* Süre

bilgilerini girebilir.

Örneğin:

```text
Semptom: headache
Şiddet: 2
Süre: 3 gün
```

Bu bilgiler model girişinde türetilmiş özellikler olarak kullanılabilir.

---

##Açıklanabilir Yapay Zekâ

Model tahminlerinin yalnızca sonuç olarak verilmesi yerine, tahmini etkileyen özelliklerin anlaşılabilmesi için:

* **SHAP**
* **LIME**

kullanılmaktadır.

Bu sayede:

```text
Neden bu hastalık tahmin edildi?
Hangi semptomlar tahmini daha fazla etkiledi?
```

gibi sorulara yönelik açıklamalar üretilebilir.

---

##Model Değerlendirme

Model performanslarını değerlendirmek için:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Cross-validation
* Sınıf bazlı metrikler

kullanılmaktadır.

Ayrıca Naive Bayes modeli için:

* Brier Score
* Expected Calibration Error (ECE)
* Reliability Diagram

gibi kalibrasyon ölçümleri de bulunmaktadır.

---

##Veri Gizliliği

Sistem **yerel olarak çalışacak şekilde tasarlanmıştır.**

Tahminler için kullanılan veriler herhangi bir uzak sunucuya gönderilmez.

Yerel telemetri mekanizması tahminleri:

```text
data/telemetry/
```

klasöründe tutabilir.

> Gerçek kişilere ait hassas veya tanımlayıcı sağlık verilerinin projeye girilmemesi önerilir.

---

#  Proje Yapısı

```text
apriori-hastalik-tespiti/
├── models/                    # eğitilmiş modeller + metadata (decision_tree, naive_bayes, ...)
│   └── manifest.json          # model versiyonlama / SHA-256 manifesti
├── data/
│   ├── Synthetic.csv          # otomatik üretilen sınıf dengeli sentetik veri
│   ├── synthetic_dataset.csv
│   ├── content/                # chatbot sözlükleri (JSON)
│   │   ├── extra_symptom_aliases.json
│   │   ├── symptom_advice.json
│   │   ├── symptom_descriptions.json
│   │   └── turkish_disease_aliases.json
│   └── telemetry/              # yerel tahmin kayıtları
├── assets/
│   └── style.css               # arayüz stilleri (app.py buradan okur)
├── src/
│   ├── preprocess.py
│   ├── split.py                # leak önleyici, grup bilinçli train/test ayrımı
│   ├── active_elicitation.py   # adaptif "sıradaki belirti" önerisi (mutual information)
│   ├── synthetic_data.py       # sınıf dengeli sentetik veri üretici
│   ├── apriori_rules.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── logistic_regression.py
│   ├── svm.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── icd10.py                # ICD-10 kod eşlemesi
│   ├── red_flags.py            # kırmızı bayrak kuralları (tek + kombinasyon)
│   ├── evaluation.py           # grup bilinçli CV, sınıf metrikleri, kalibrasyon
│   ├── model_metadata.py       # veri parmak izi, model durumu, toplu yeniden eğitim
│   ├── versioning.py           # manifest + SHA-256 + schema versiyonu
│   ├── telemetry.py            # yerel tahmin kayıtları ve ayrılık özeti
│   ├── reports.py              # txt/HTML/PDF rapor üretimi
│   ├── predict.py
│   ├── visualization.py
│   ├── explainability.py
│   ├── chatbot.py
│   ├── llm.py                  # opsiyonel LLM entegrasyonu (Ollama/OpenAI/Claude)
│   ├── i18n.py                 # Türkçe/İngilizce çeviri sözlüğü
│   ├── disease_info.py
│   ├── model_metadata.py
│   └── utils.py
├── tests/                      # pytest testleri
├── tools/                      # yardımcı betikler (retrain_all_models.py, eval_check.py, ...)
├── app.py                      # Streamlit arayüzü
├── api.py                      # FastAPI tabanlı REST API
├── training_duzenlenmis.csv    # ana eğitim veri kümesi
├── run.bat                     # Windows: tek tıkla çalıştırma
├── run.sh                      # macOS/Linux: tek tıkla çalıştırma
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── Dockerfile
└── README.md
```

---

# ⚙️ Kurulum

## Gereksinimler

* Python **3.12+**
* pip
* Git
* İsteğe bağlı: Docker

---

---

## 3. Gerekli Paketleri Yükleme

```bash
pip install -r requirements.txt
```

Geliştirme ve test bağımlılıkları için:

```bash
pip install -r requirements-dev.txt
```

---

#Uygulamayı Çalıştırma

## Windows

Projede bulunan:

```text
run.bat
```

dosyasına çift tıklayabilirsiniz.

Alternatif olarak:

```bash
streamlit run app.py
```

---

## macOS / Linux

```bash
chmod +x run.sh
./run.sh
```

veya:

```bash
streamlit run app.py
```

Uygulama varsayılan olarak:

```text
http://localhost:8501
```

adresinde çalışır.

---

# Model Versiyonlama

Model artefaktlarının bütünlüğünü takip etmek için:

* `model_schema_version`
* SHA-256
* Veri fingerprint
* Eğitim zamanı
* Veri satır sayısı
* Sınıf sayısı

bilgileri manifest içerisinde tutulmaktadır.

Örnek:

```text
models/
└── manifest.json
```

Bu yapı, model dosyalarının hangi veri setinden üretildiğinin takip edilmesini kolaylaştırır.

---

# Kullanılan Teknolojiler

### Programlama

* Python

### Veri İşleme

* Pandas
* NumPy

### Makine Öğrenmesi

* Scikit-learn
* XGBoost
* LightGBM

### Birliktelik Analizi

* MLxtend / Apriori

### Açıklanabilir Yapay Zekâ

* SHAP
* LIME

### Arayüz

* Streamlit

### API

* FastAPI
* Uvicorn
* Pydantic

### Görselleştirme

* Matplotlib

### Model Kaydetme

* Joblib

### Test

* Pytest

### Container

* Docker

---

# Veri Seti

Proje eğitim aşamasında hastalık ve semptom ilişkilerini içeren veri setlerinden yararlanır.

Mevcut model manifestine göre kullanılan temel eğitim yapılandırmasında:

```text
Eğitim verisi : Training.csv
Satır sayısı  : 298
Hastalık sınıfı: 40
```


# Model Pipeline

Genel tahmin pipeline'ı:

```text
1. Veri yükleme
        ↓
2. Veri ön işleme
        ↓
3. Semptomların normalize edilmesi
        ↓
4. Model girdisinin hazırlanması
        ↓
5. 7 farklı model ile tahmin
        ↓
6. Model sonuçlarının ağırlıklandırılması
        ↓
7. Ensemble tahmin
        ↓
8. Güven seviyesi
        ↓
9. OOD kontrolü
        ↓
10. Red flag kontrolü
        ↓
11. ICD-10 eşlemesi
        ↓
12. Açıklanabilirlik
        ↓
13. Raporlama
```

---

#  Raporlama

Sistem tahmin sonuçlarının raporlanmasını destekler.

Raporlarda kullanılabilecek bilgiler:

* Tahmin edilen hastalık
* ICD-10 kodu
* Olasılık sıralaması
* Model sonuçları
* Model doğrulukları
* Ayırıcı tanılar
* Semptom bilgileri

Ayrıca çıktıların farklı formatlarda oluşturulması için raporlama modülü bulunmaktadır.

---


#  Projenin Teknik Özeti

| Bileşen             | Kullanılan Yöntem           |
| ------------------- | --------------------------- |
| Hastalık tahmini    | 7 ML modeli                 |
| Ensemble            | Weighted Soft Voting        |
| Birliktelik analizi | Apriori                     |
| Adaptif soru        | Mutual Information          |
| Açıklanabilirlik    | SHAP + LIME                 |
| Güven analizi       | Naive Bayes olasılıkları    |
| OOD                 | Jaccard tabanlı kontrol     |
| Kritik durum        | Red Flag kuralları          |
| Kodlama             | ICD-10                      |
| Arayüz              | Streamlit                   |
| API                 | FastAPI                     |
| Raporlama           | TXT / HTML / PDF            |
| Test                | Pytest                      |
| Container           | Docker                      |
| Model bütünlüğü     | SHA-256                     |
| Model takibi        | Manifest + Data Fingerprint |

---

#  Güvenlik ve Etik Kullanım

Bu proje eğitim, araştırma ve karar destek amacıyla geliştirilmiştir.

Sistem:

* Kesin tıbbi tanı koymaz.
* Doktor muayenesinin yerine geçmez.
* Gerçek hasta verileriyle klinik kullanım için doğrulanmış bir tıbbi cihaz olarak değerlendirilmemelidir.
* Acil durumlarda sağlık kuruluşlarına başvurulmalıdır.

Gerçek kişilere ait kimlik bilgileri veya hassas sağlık verilerinin sisteme girilmemesi önerilir.

---

# Gelecekte Geliştirilebilecek Alanlar

Projeye ilerleyen aşamalarda aşağıdaki özellikler eklenebilir:

* Daha büyük ve gerçek dünya klinik veri setleri
* Daha gelişmiş kalibrasyon yöntemleri
* Transformer tabanlı semptom sınıflandırma
* Daha gelişmiş çoklu hastalık tahmini
* Zaman serisi tabanlı hasta takibi
* Kullanıcı bazlı geçmiş tahmin analizi
* Gelişmiş model izleme sistemi
* Web tabanlı deployment
* Kullanıcı ve yetkilendirme sistemi
* Klinik veri standartlarıyla daha ileri entegrasyon

---

#  Geliştirici

**Fahri Kantaş**

**Salih Emre Kesici**

---

#  Lisans

Bu projenin lisans koşulları için repository içerisindeki lisans dosyasını inceleyiniz.

---

##  Disclaimer

**Bu yazılım yalnızca eğitim, araştırma ve karar destek amacıyla geliştirilmiştir.**

Üretilen tahminler tıbbi tanı olarak değerlendirilmemelidir. Herhangi bir sağlık problemi, acil durum veya tedavi kararı için yetkili bir sağlık profesyoneline başvurulmalıdır.
