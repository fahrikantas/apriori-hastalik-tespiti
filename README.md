# 🩺 Semptom Verilerine Dayalı Hastalık Tahmin ve Karar Destek Sistemi

Semptom verilerinden olası hastalıkları tahmin etmek amacıyla geliştirilen, **makine öğrenmesi, birliktelik analizi, topluluk öğrenmesi ve açıklanabilir yapay zekâ** yöntemlerini bir araya getiren modüler bir karar destek sistemidir.

Sistem; kullanıcının seçtiği semptomları analiz ederek birden fazla makine öğrenmesi modelinden tahminler üretir, modellerin sonuçlarını **ağırlıklı soft-voting** yöntemiyle birleştirir ve sonuçları güven seviyesi, ICD-10 kodu, ayırıcı tanılar, Apriori kuralları ve açıklanabilirlik çıktılarıyla birlikte sunar.

> ⚠️ **Önemli:** Bu proje tıbbi karar destek amacıyla geliştirilmiştir ve profesyonel tıbbi tanının yerine geçmez. Gerçek sağlık sorunlarında mutlaka bir sağlık profesyoneline başvurulmalıdır.

---

## 📌 Projenin Amacı

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

# ✨ Özellikler

## 🤖 Çoklu Makine Öğrenmesi

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

## 🗳️ Ağırlıklı Soft-Voting Ensemble

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

## 🔗 Apriori Birliktelik Kuralları

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

## 🧠 Adaptif Belirti Toplama

Sistem yalnızca kullanıcının verdiği semptomlarla sınırlı kalmaz.

**"Sıradaki belirtiyi sor"** mekanizması ile henüz seçilmemiş semptomlar arasından bilgi kazancı yüksek olan semptom önerilebilir.

Bu mekanizma:

* Mutual Information
* Mevcut semptomlar
* Önceki tahminler

gibi bilgilerden yararlanarak daha odaklı semptom toplama süreci oluşturur.

---

## 📊 Hastalık Tahmini ve Ayırıcı Tanı

Sistem yalnızca tek bir hastalık döndürmek yerine birden fazla olası hastalığı değerlendirebilir.

Çıktılarda:

* Nihai tahmin
* En olası hastalıklar
* Top-5 ayırıcı tanı
* Her tanıyı destekleyen modeller
* Model skorları

gösterilebilir.

---

## 🚦 Güven Seviyesi

Tahmin sonucuna göre güven seviyesi oluşturulur:

| Seviye    | Açıklama     |
| --------- | ------------ |
| 🟢 High   | Yüksek güven |
| 🟡 Medium | Orta güven   |
| 🔴 Low    | Düşük güven  |

Güven seviyesi özellikle modellerin birbirinden farklı sonuçlar ürettiği durumlarda kullanıcıya ek bilgi sağlar.

---


## 🔍 OOD (Out-of-Distribution) Algılama

Girilen semptom kombinasyonunun eğitim verilerine ne kadar benzediği kontrol edilir.

Sistem, eğitim verisinden oldukça farklı bir semptom kombinasyonu tespit ettiğinde kullanıcıya **OOD uyarısı** verebilir.

Bu özellik, modelin eğitim verisinin dışında kalan örneklerde aşırı güvenli sonuçlar üretmesini önlemeye yardımcı olur.

---

## 🚨 Red Flag (Kritik Belirti) Kontrolü

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

## 🏥 ICD-10 Entegrasyonu

Tahmin edilen hastalıklar **ICD-10** kodlarıyla eşleştirilir.

ICD-10 kodları:

* Tahmin sonuçlarında
* Ayırıcı tanılarda
* Raporlarda

gösterilebilir.

Projede hastalık etiketleri için kapsamlı ICD-10 eşlemesi bulunmaktadır.

---

## 💬 Türkçe Sohbet Asistanı

Projede semptom ve hastalık bilgileriyle etkileşim kurulabilmesini sağlayan Türkçe sohbet asistanı bulunmaktadır.

Sistem:

* Serbest metin semptomlarını algılayabilir
* Türkçe semptom adlarını destekler
* Semptom/hastalık bilgileri sunabilir
* Konuşma bağlamını takip edebilir
* Kullanıcı geri bildirimi alabilir

---

## 🌍 Türkçe / İngilizce Dil Desteği

Arayüz iki dili destekler:

* 🇹🇷 Türkçe
* 🇬🇧 English

Çeviri metinleri `src/i18n.py` içerisinde merkezi olarak yönetilir.

---

## 🧩 Semptom Şiddeti ve Süresi

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

## 🔬 Açıklanabilir Yapay Zekâ

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

## 📈 Model Değerlendirme

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

## 🔐 Veri Gizliliği

Sistem **yerel olarak çalışacak şekilde tasarlanmıştır.**

Tahminler için kullanılan veriler herhangi bir uzak sunucuya gönderilmez.

Yerel telemetri mekanizması tahminleri:

```text
data/telemetry/
```

klasöründe tutabilir.

> Gerçek kişilere ait hassas veya tanımlayıcı sağlık verilerinin projeye girilmemesi önerilir.

---

# 🗂️ Proje Yapısı

```text
DiseasePrediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── assets/
│   └── style.css
│
├── data/
│   ├── content/
│   │   ├── extra_symptom_aliases.json
│   │   ├── symptom_advice.json
│   │   ├── symptom_descriptions.json
│   │   └── turkish_disease_aliases.json
│   │
│   ├── telemetry/
│   │   └── predictions.jsonl
│   │
│   ├── Synthetic.csv
│   └── synthetic_dataset.csv
│
├── models/
│   ├── apriori_rules.pkl
│   ├── decision_tree.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   └── manifest.json
│
├── src/
│   ├── active_elicitation.py
│   ├── apriori_rules.py
│   ├── chatbot.py
│   ├── decision_tree.py
│   ├── disease_info.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── i18n.py
│   ├── icd10.py
│   ├── lightgbm_model.py
│   ├── llm.py
│   ├── logistic_regression.py
│   ├── model_metadata.py
│   ├── naive_bayes.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── random_forest.py
│   ├── red_flags.py
│   ├── reports.py
│   ├── split.py
│   ├── svm.py
│   ├── synthetic_data.py
│   ├── telemetry.py
│   ├── utils.py
│   ├── versioning.py
│   ├── visualization.py
│   └── xgboost_model.py
│
├── tests/
│   ├── test_api.py
│   ├── test_app_flow.py
│   ├── test_calibration.py
│   ├── test_chatbot.py
│   ├── test_content.py
│   ├── test_differential.py
│   ├── test_evaluation.py
│   ├── test_icd10.py
│   ├── test_llm.py
│   ├── test_ood.py
│   ├── test_pipeline.py
│   ├── test_red_flags.py
│   ├── test_retraining.py
│   ├── test_split.py
│   ├── test_synthetic.py
│   ├── test_telemetry.py
│   └── test_versioning.py
│
├── api.py
├── app.py
├── Dockerfile
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
├── run.bat
├── run.sh
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

## 1. Repoyu Klonlama

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
```

---

## 2. Sanal Ortam Oluşturma

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

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

# ▶️ Uygulamayı Çalıştırma

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

# 🔌 REST API

Proje, Streamlit arayüzünün yanında **FastAPI tabanlı REST API** de sunmaktadır.

API'yi başlatmak için:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Geliştirme sırasında otomatik yeniden yükleme:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpointleri

| Method | Endpoint        | Açıklama                    |
| ------ | --------------- | --------------------------- |
| `GET`  | `/`             | API genel bilgileri         |
| `GET`  | `/health`       | Sistem ve model durumu      |
| `GET`  | `/datasets`     | Kullanılabilir veri setleri |
| `GET`  | `/api/symptoms` | Desteklenen semptomlar      |
| `POST` | `/api/predict`  | Hastalık tahmini            |

FastAPI'nin otomatik dokümantasyonuna:

```text
http://localhost:8000/docs
```

adresinden ulaşılabilir.

---

## Örnek API İsteği

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"symptoms\":[\"itching\",\"skin_rash\"]}"
```

Örnek yapı:

```json
{
  "symptoms": [
    "itching",
    "skin_rash"
  ],
  "training_file": "Training.csv"
}
```

API yanıtında model tahminleriyle birlikte:

* Nihai tahmin
* ICD-10 kodu
* Güven seviyesi
* Model bazlı tahminler
* Apriori sonuçları
* OOD durumu
* Model anlaşması

gibi bilgiler bulunabilir.

---

# 🐳 Docker ile Çalıştırma

Docker image oluşturmak için:

```bash
docker build -t hastalik-tespiti .
```

Container'ı çalıştırmak için:

```bash
docker run -p 8501:8501 hastalik-tespiti
```

Daha sonra:

```text
http://localhost:8501
```

adresinden uygulamaya erişilebilir.

---


# 🧬 Model Versiyonlama

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

# 📦 Kullanılan Teknolojiler

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

# 📊 Veri Seti

Proje eğitim aşamasında hastalık ve semptom ilişkilerini içeren veri setlerinden yararlanır.

Mevcut model manifestine göre kullanılan temel eğitim yapılandırmasında:

```text
Eğitim verisi : Training.csv
Satır sayısı  : 298
Hastalık sınıfı: 40
```

Ayrıca proje içerisinde sınıf dengeli sentetik veri üretimi ve `Synthetic.csv` veri seti desteği bulunmaktadır.

---

# 🧪 Model Pipeline

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

# 📄 Raporlama

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

# 🤖 Opsiyonel LLM Desteği

Projede isteğe bağlı olarak LLM entegrasyonu bulunmaktadır.

Desteklenen yaklaşımlar:

* Ollama
* OpenAI
* Anthropic / Claude

LLM'nin kullanım amacı **nihai hastalık tanısı koymak değildir**.

LLM yalnızca serbest metinden semptom kodlarının çıkarılması gibi yardımcı işlemlerde kullanılabilir.

Nihai hastalık tahmini:

```text
LLM
 │
 └──► Semptom çıkarımı
          │
          ▼
     Kural tabanlı doğrulama
          │
          ▼
   Makine öğrenmesi modelleri
          │
          ▼
    Nihai tahmin
```

Bu tasarımda LLM'nin doğrudan tanı üretmesi engellenmiştir.

---

# 📌 Projenin Teknik Özeti

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

# 🔒 Güvenlik ve Etik Kullanım

Bu proje eğitim, araştırma ve karar destek amacıyla geliştirilmiştir.

Sistem:

* Kesin tıbbi tanı koymaz.
* Doktor muayenesinin yerine geçmez.
* Gerçek hasta verileriyle klinik kullanım için doğrulanmış bir tıbbi cihaz olarak değerlendirilmemelidir.
* Acil durumlarda sağlık kuruluşlarına başvurulmalıdır.

Gerçek kişilere ait kimlik bilgileri veya hassas sağlık verilerinin sisteme girilmemesi önerilir.

---

# 🚀 Gelecekte Geliştirilebilecek Alanlar

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

# 👨‍💻 Geliştirici

**Fahri Kantaş**
**Salih Emre Kesici**

---

# 📜 Lisans

Bu projenin lisans koşulları için repository içerisindeki lisans dosyasını inceleyiniz.

---

## ⚠️ Disclaimer

**Bu yazılım yalnızca eğitim, araştırma ve karar destek amacıyla geliştirilmiştir.**

Üretilen tahminler tıbbi tanı olarak değerlendirilmemelidir. Herhangi bir sağlık problemi, acil durum veya tedavi kararı için yetkili bir sağlık profesyoneline başvurulmalıdır.
