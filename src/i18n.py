"""Minimal two-language UI translation helper (Turkish / English)."""

from __future__ import annotations

LANGUAGES: dict[str, str] = {
    "tr": "Türkçe",
    "en": "English",
}
DEFAULT_LANGUAGE = "tr"

_TRANSLATIONS: dict[str, dict[str, str]] = {
    "app_title": {"tr": "Semptoma Dayalı Hastalık Tahmin ve Karar Destek Sistemi", "en": "Symptom-Based Disease Prediction and Decision Support System"},
    "app_subtitle": {
        "tr": "Semptomları seçersiniz; sistem arka planda Apriori, Decision Tree, Naive Bayes, Random Forest, Logistic Regression, SVM, XGBoost ve LightGBM analizlerini birlikte çalıştırır.",
        "en": "Select symptoms; the system jointly runs Apriori, Decision Tree, Naive Bayes, Random Forest, Logistic Regression, SVM, XGBoost and LightGBM analyses.",
    },
    "app_short_name": {"tr": "Hastalık Tahmin", "en": "Disease Predictor"},
    "app_brand_sub": {"tr": "Tıbbi Karar Destek", "en": "Clinical Decision Support"},
    "app_badge": {"tr": "Yapay Zeka Destekli", "en": "AI Powered"},
    "hero_step1_title": {"tr": "Belirtileri Seçin", "en": "Select Symptoms"},
    "hero_step1_text": {"tr": "Ateş, öksürük, baş ağrısı...", "en": "Fever, cough, headache..."},
    "hero_step2_title": {"tr": "Analiz Edin", "en": "Run Analysis"},
    "hero_step2_text": {"tr": "8 makine öğrenmesi modeli birlikte çalışır", "en": "8 ML models run jointly"},
    "hero_step3_title": {"tr": "Sonucu Yorumlayın", "en": "Interpret Result"},
    "hero_step3_text": {"tr": "Güven skoru ve ayırıcı tanı listesiyle", "en": "With confidence score & differential list"},
    "result_card_label": {"tr": "Tahmini Hastalık", "en": "Predicted Disease"},
    "result_card_note": {"tr": "Karar destek amaçlıdır; tıbbi tanı koymaz.", "en": "For decision support only."},
    "confidence_bar_label": {"tr": "Model Güveni", "en": "Model Confidence"},
    "confidence_bar_detail": {"tr": "En yüksek olasılık: {}", "en": "Highest probability: {}"},
    "stat_symptoms": {"tr": "Seçili Belirti", "en": "Selected Symptoms"},
    "stat_rules": {"tr": "Apriori Kuralı", "en": "Apriori Rules"},
    "stat_models": {"tr": "Model", "en": "Models"},
    "stat_diseases": {"tr": "Hastalık", "en": "Diseases"},
    "top_conditions_title": {"tr": "En Olası Hastalıklar", "en": "Top Conditions"},
    "top_conditions_caption": {"tr": "Naive Bayes olasılıklarına göre ilk 3", "en": "Top 3 by Naive Bayes probability"},
    "agreement_label": {"tr": "Model Uyumu", "en": "Model Agreement"},
    "downloads_title": {"tr": "Rapor İndir", "en": "Download Report"},
    "downloads_caption": {"tr": "Analiz sonucunu dışa aktar", "en": "Export the analysis result"},
    "language": {"tr": "Dil", "en": "Language"},
    "symptom_selection": {"tr": "Belirti Seçimi", "en": "Symptom Selection"},
    "symptom_search": {"tr": "Semptom ara", "en": "Search symptoms"},
    "symptom_search_placeholder": {"tr": "Ara: ateş, öksürük, baş ağrısı...", "en": "Search: fever, cough, headache..."},
    "symptoms_select": {"tr": "Semptomlarınızı seçiniz", "en": "Select symptoms"},
    "symptoms_select_hint": {"tr": "Seçmek için yazın...", "en": "Type to select..."},
    "analyze": {"tr": "Analiz Et", "en": "Analyze"},
    "reset": {"tr": "Sıfırla", "en": "Reset"},
    "no_match": {"tr": "Arama için eşleşen belirti bulunamadı.", "en": "No symptom matched your search."},
    "select_first": {"tr": "Soldaki menüden belirtileri seçip Analiz Et'ye basın.", "en": "Select symptoms from the sidebar and press Analyze."},
    "select_at_least_one": {"tr": "Lütfen en az bir belirti seçin.", "en": "Please select at least one symptom."},
    "running_models": {"tr": "Modeller çalıştırılıyor...", "en": "Running models..."},
    "data_visualizations": {"tr": "Veri Görselleştirmeleri", "en": "Data Visualizations"},
    "sidebar_search_hint": {"tr": "Belirti listesini filtrelemek için arama kutusunu kullanın.", "en": "Use the sidebar search box to filter the symptom list."},
    "final_result": {"tr": "Sonuç", "en": "Final Result"},
    "predicted": {"tr": "Tahmini hastalık: {}", "en": "Predicted disease: {}"},
    "decision_support_only": {"tr": "Bu sistem yalnızca karar destek amaçlıdır; tıbbi tanı yerine geçmez.", "en": "This system is for decision support only. It does not replace a medical diagnosis."},
    "confidence_level": {"tr": "Güven seviyesi:", "en": "Confidence level:"},
    "high": {"tr": "Yüksek", "en": "High"},
    "medium": {"tr": "Orta", "en": "Medium"},
    "low": {"tr": "Düşük", "en": "Low"},
    "ambiguous_warning": {"tr": "⚠️ Modeller farklı sonuç verdi; bu sonuç **belirsiz** olarak değerlendirilmelidir. Ek belirti ekleyip yeniden deneyin.", "en": "⚠️ The models disagree; treat this result as **ambiguous**. Add more symptoms and retry."},
    "low_confidence_warning": {"tr": "⚠️ Model güveni düşük; sonuç dikkatli yorumlanmalıdır.", "en": "⚠️ Model confidence is low; interpret the result with caution."},
    "top3": {"tr": "**En olası 3 hastalık (Naive Bayes):**", "en": "**Top 3 likely diseases (Naive Bayes):**"},
    "model_predictions": {"tr": "Model Tahminleri", "en": "Model Predictions"},
    "download_txt": {"tr": "📄 Raporu İndir (.txt)", "en": "📄 Download Report (.txt)"},
    "download_html": {"tr": "🖨️ Yazdırılabilir Rapor (.html)", "en": "🖨️ Printable Report (.html)"},
    "apriori_results": {"tr": "Apriori Sonuçları", "en": "Apriori Results"},
    "no_apriori_rule": {"tr": "Seçilen belirtiler için güçlü bir Apriori kuralı bulunamadı.", "en": "No strong Apriori rule was found for the selected symptoms."},
    "dt_visualization": {"tr": "Decision Tree Görselleştirmesi", "en": "Decision Tree Visualization"},
    "nb_probabilities": {"tr": "Naive Bayes Olasılıkları", "en": "Naive Bayes Probabilities"},
    "feature_importance": {"tr": "Özellik Önem Düzeyleri", "en": "Feature Importance"},
    "model_accuracy": {"tr": "Model Doğruluğu", "en": "Model Accuracy"},
    "accuracy_caption": {"tr": "Doğrulamalar 80/20 eğitim/doğrulama ayrımının test kısmında hesaplanmıştır; eğitim verisi üzerinden değildir.", "en": "Scores are computed on the test slice of an 80/20 train/validation split, not on training data."},
    "advanced_eval": {"tr": "Gelişmiş Değerlendirme", "en": "Advanced Evaluation"},
    "cv_caption": {"tr": "5 katlı çapraz doğrulama (StratifiedKFold) ile model doğruluğu — tek bir sabit ayrımdan daha güvenilir.", "en": "Model accuracy via 5-fold stratified cross-validation — more reliable than a single fixed split."},
    "cv_running": {"tr": "Çapraz doğrulama hesaplanıyor...", "en": "Computing cross-validation..."},
    "eval_model_select": {"tr": "Sınıf bazlı metrikler için model seçin", "en": "Choose a model for class-level metrics"},
    "per_class_metrics": {"tr": "Sınıf Bazlı Metrikler", "en": "Per-Class Metrics"},
    "confusion_matrix": {"tr": "Karışıklık Matrisi", "en": "Confusion Matrix"},
    "privacy": {"tr": "🔒 Gizlilik: Uygulama tamamen yerel çalışır; seçtiğiniz belirtiler ve sohbet hiçbir sunucuya gönderilmez.", "en": "🔒 Privacy: the app runs fully locally; your symptoms and chat never leave your machine."},
    "retrain_button": {"tr": "🔄 Modelleri Yeniden Eğit", "en": "🔄 Retrain Models"},
    "retrain_spinner": {"tr": "Modeller yeniden eğitiliyor...", "en": "Retraining models..."},
    "retrain_done": {"tr": "Modeller başarıyla yeniden eğitildi.", "en": "Models retrained successfully."},
    "stale_warning": {"tr": "⚠️ Eğitim verisi değişmiş; modeller güncel değil. Lütfen 'Modelleri Yeniden Eğit' butonunu kullanın.", "en": "⚠️ The training data has changed; models are outdated. Use the 'Retrain Models' button."},
    "model_status": {"tr": "Model Durumu", "en": "Model Status"},
    "data_health": {"tr": "Veri Sağlığı", "en": "Data Health"},
    "data_health_ok": {"tr": "Veri dosyası yüklendi: {} satır, {} belirti, {} sınıf.", "en": "Data loaded: {} rows, {} symptoms, {} classes."},
    "data_missing": {"tr": "❌ Eğitim verisi bulunamadı (Training.csv).", "en": "❌ Training data not found (Training.csv)."},
    "fresh": {"tr": "Güncel", "en": "Up to date"},
    "stale": {"tr": "Eski", "en": "Stale"},
    "missing": {"tr": "Eksik", "en": "Missing"},
    "chat_export": {"tr": "Sohbet Dökümü", "en": "Chat Export"},
    "tab_prediction": {"tr": "Tahmin", "en": "Prediction"},
    "tab_visualization": {"tr": "Görselleştirme", "en": "Visualization"},
    "tab_evaluation": {"tr": "Değerlendirme", "en": "Evaluation"},
    "tab_explainability": {"tr": "Açıklanabilirlik", "en": "Explainability"},
    "tab_disease_info": {"tr": "Hastalık Bilgisi", "en": "Disease Info"},
    "tab_chat": {"tr": "Sohbet", "en": "Chat"},
    "download_pdf": {"tr": "📑 Raporu İndir (.pdf)", "en": "📑 Download Report (.pdf)"},
    "apriori_settings": {"tr": "Apriori Parametreleri", "en": "Apriori Parameters"},
    "apriori_min_support": {"tr": "Minimum destek", "en": "Minimum support"},
    "apriori_min_confidence": {"tr": "Minimum güven", "en": "Minimum confidence"},
    "apriori_min_lift": {"tr": "Minimum lift", "en": "Minimum lift"},
    "apriori_max_len": {"tr": "Maksimum kural uzunluğu", "en": "Maximum rule length"},
    "llm_settings": {"tr": "LLM Ayarları", "en": "LLM Settings"},
    "llm_provider": {"tr": "Sağlayıcı", "en": "Provider"},
    "llm_off": {"tr": "Kapalı (kural tabanlı)", "en": "Off (rule-based)"},
    "llm_ollama": {"tr": "Ollama (yerel)", "en": "Ollama (local)"},
    "llm_openai": {"tr": "OpenAI", "en": "OpenAI"},
    "llm_anthropic": {"tr": "Anthropic (Claude)", "en": "Anthropic (Claude)"},
    "llm_model": {"tr": "Model adı", "en": "Model name"},
    "llm_api_key": {"tr": "API anahtarı", "en": "API key"},
    "llm_anthropic_api_key": {"tr": "Anthropic API anahtarı", "en": "Anthropic API key"},
    "llm_url": {"tr": "Ollama URL", "en": "Ollama URL"},
    "llm_active": {"tr": "LLM asistanı aktif", "en": "LLM assistant active"},
    "llm_fallback": {"tr": "LLM yanıt veremedi; kural tabanlı asistana geçildi.", "en": "LLM unavailable; switched to rule-based assistant."},
    "chat_title": {"tr": "Sohbet Asistanı", "en": "Chat Assistant"},
    "chat_caption": {"tr": "Bu asistan karar destek amaçlıdır; tıbbi tanı koymaz.", "en": "This assistant is for decision support only; it does not diagnose."},
    "chat_clear": {"tr": "🔄 Sohbeti Temizle", "en": "🔄 Clear Chat"},
    "chat_input_placeholder": {"tr": "Asistan'a yazın... (örnek: \"Dengue nedir?\", \"ateşim ve öksürüğüm var\")", "en": "Write to the assistant... (e.g. \"What is Dengue?\", \"I have fever and cough\")"},
    "chat_feedback_question": {"tr": "Bu yanıtlar işine yaradı mı?", "en": "Were these answers helpful?"},
    "chat_feedback_positive": {"tr": "👍 Yararlı", "en": "👍 Helpful"},
    "chat_feedback_negative": {"tr": "👎 Yararlı değil", "en": "👎 Not helpful"},
    "chat_feedback_thanks": {"tr": "Geri bildiriminiz için teşekkürler! 😊", "en": "Thanks for your feedback! 😊"},
    "chat_feedback_note": {"tr": "Anlaşıldı; bu tür soruları daha iyi yanıtlamak için geliştirmeye devam ediyoruz.", "en": "Understood; we keep improving answers like these."},
    "chat_download": {"tr": "💬 Sohbeti indir", "en": "💬 Download chat"},
    "chat_selected": {"tr": "Seçili", "en": "Selected"},
    "chat_result": {"tr": "Sonuç", "en": "Result"},
    "chat_welcome": {
        "tr": (
            "Merhaba! 👋 Ben bu arayüzün sohbet asistanıyım.\n\n"
            "Bana şunları sorabilirsin:\n"
            "- **Sonucu açıkla** — model sonuçlarını yorumlarım\n"
            "- **Apriori kuralını yorumla** — en güçlü birliktelik kuralını anlatırım\n"
            "- **Ne yapmalıyım?** — seçili belirtilere göre öneri veririm\n"
            "- **Dengue nedir?** / **Ateş hangi hastalıklarda görülür?** — hastalık ve belirti bilgisi veririm\n\n"
            "Belirtilerinizi yazarak da deneyebilirsiniz: \"ateşim ve öksürüğüm var\"."
        ),
        "en": (
            "Hello! 👋 I am the chat assistant for this interface.\n\n"
            "You can ask me to:\n"
            "- **Explain the result** — interpret model outputs\n"
            "- **Explain the Apriori rule** — describe the strongest association rule\n"
            "- **What should I do?** — suggestions based on selected symptoms\n"
            "- **What is Dengue?** / **Which diseases include fever?** — disease and symptom info\n\n"
            "You can also type your symptoms directly: \"I have fever and cough\"."
        ),
    },
    "chat_quick_explain": {"tr": "Sonucu açıkla", "en": "Explain result"},
    "chat_quick_apriori": {"tr": "Apriori kuralını yorumla", "en": "Explain Apriori rule"},
    "chat_quick_next": {"tr": "Ne yapmalıyım?", "en": "What should I do?"},
    "chat_quick_eval": {"tr": "Hastalıklarımı değerlendir", "en": "Evaluate my diseases"},
    "chat_quick_accuracy": {"tr": "Model doğruluğu", "en": "Model accuracy"},
    "chat_quick_how": {"tr": "Sistem nasıl çalışıyor?", "en": "How does the system work?"},
    "no_symptoms_selected": {"tr": "Henüz semptom seçilmedi.", "en": "No symptoms selected yet."},
    "explainability_title": {"tr": "Model Açıklanabilirliği", "en": "Model Explainability"},
    "explainability_caption": {"tr": "", "en": "SHAP for tree models; LIME explains a single prediction for any model."},
    "explain_model_select": {"tr": "Açıklama için model seçin", "en": "Choose a model to explain"},
    "shap_chart": {"tr": "SHAP Grafiği", "en": "SHAP Chart"},
    "lime_chart": {"tr": "LIME Grafiği", "en": "LIME Chart"},
    "disease_info_title": {"tr": "Hastalık Bilgi Kartları", "en": "Disease Information Cards"},
    "disease_info_caption": {"tr": "Veri seti istatistikleri ve kısa açıklamalar.", "en": "Dataset statistics and short descriptions."},
    "disease_select": {"tr": "Hastalık seçin", "en": "Select a disease"},
    "disease_prevalence": {"tr": "Veri setindeki sıklık", "en": "Dataset prevalence"},
    "disease_records": {"tr": "Kayıt sayısı", "en": "Record count"},
    "disease_top_symptoms": {"tr": "En sık görülen belirtiler", "en": "Most common symptoms"},
    "disease_when_doctor": {"tr": "Ne zaman doktora gidilmeli?", "en": "When to see a doctor?"},
    "predicted_disease_card": {"tr": "Tahmin edilen hastalık kartı", "en": "Predicted disease card"},
    "dataset": {"tr": "Veri Seti", "en": "Dataset"},
    "dataset_default": {"tr": "Orijinal (training_duzenlenmis)", "en": "Original (training_duzenlenmis)"},
    "dataset_synthetic": {"tr": "Sentetik (daha zengin, 5000 satır)", "en": "Synthetic (richer, 5000 rows)"},
    "dataset_loading": {"tr": "Veri seti hazırlanıyor; modeller güncelleniyor...", "en": "Preparing dataset; updating models..."},
    "leak_fix_caption": {
        "tr": (
            "🔬 Doğruluklar grup bilinçli (sızıntı önleyici) ayrımla hesaplanır: aynı veya neredeyse "
            "aynı belirti desenleri eğitim ve test taraflarına ayrıştırılmaz, böylece model desenleri "
            "ezberleyerek şişirilmiş skor üretemez. Bu sürüm 4.920 satır ve her hastalık için birden "
            "çok varyasyon içerdiğinden dürüst doğruluklar yüksektir; 'Synthetic.csv' kenar çubuğundan "
            "daha zengin bir varyant üretir."
        ),
        "en": (
            "🔬 Scores are computed with a group-aware (leak-preventing) split: identical or "
            "near-identical symptom patterns never straddle the train/test boundary, so models "
            "cannot memorize patterns and inflate scores. This version has 4,920 rows with "
            "multiple variations per disease, so honest accuracy is high; 'Synthetic.csv' "
            "generates a richer variant from the sidebar."
        ),
    },
    "symptom_details": {"tr": "Belirti Detayları", "en": "Symptom Details"},
    "symptom_details_caption": {"tr": "Her belirti için şiddet ve süre belirtebilirsiniz.", "en": "Set severity and duration for each symptom."},
    "severity": {"tr": "Şiddet", "en": "Severity"},
    "duration_days": {"tr": "Süre (gün)", "en": "Duration (days)"},
    "suggest_next": {"tr": "💡 Sonraki belirtiyi öner", "en": "💡 Suggest next symptom"},
    "suggested_next": {"tr": "Önerilen belirti: **{}**", "en": "Suggested next symptom: **{}**"},
    "suggested_none": {"tr": "Şu an için ek bir belirti önerisi yok.", "en": "No additional symptom suggestion for now."},
    "ood_warning": {
        "tr": "⚠️ **Belirsiz kombinasyon:** Seçtiğiniz semptomların birlikte görüldüğü bir kayıt eğitim verisinde yok (en yakın eşleşme %{}). Sonuç yalnızca tahminîdir.",
        "en": "⚠️ **Out-of-distribution:** No training record contains this symptom combination (nearest overlap {}%). Treat the result as tentative.",
    },
    "ood_ok": {
        "tr": "Bu kombinasyon eğitim verisindeki kayıtlara benziyor (en yakın eşleşme %{}).",
        "en": "This combination resembles training records (nearest overlap {}%).",
    },
    "calibration": {"tr": "Kalibrasyon (Naive Bayes)", "en": "Calibration (Naive Bayes)"},
    "calibration_caption": {
        "tr": "Brier skoru ve ECE (beklenen kalibrasyon hatası) doğrulama verisinde hesaplanır; 0'a yakın değerler iyi kalibrasyon demektir (güven seviyeleri gerçek başarıyla örtüşür).",
        "en": "Brier score and ECE (expected calibration error) are computed on the holdout set; values near 0 mean well-calibrated confidence.",
    },
    "reliability_table": {"tr": "Güvenilirlik Tablosu", "en": "Reliability Table"},
    "telemetry": {"tr": "Lokal Telemetri (model anlaşmazlıkları)", "en": "Local Telemetry (model disagreements)"},
    "telemetry_privacy": {
        "tr": "🔒 Her tahmin yalnızca bu makinede data/telemetry/predictions.jsonl dosyasına kaydedilir; kişisel veri yok, hiçbir şey gönderilmez.",
        "en": "🔒 Each prediction is logged only locally in data/telemetry/predictions.jsonl; no personal data, nothing is transmitted.",
    },
    "telemetry_records": {"tr": "Kayıt", "en": "Records"},
    "telemetry_disagreements": {"tr": "Anlaşmazlık", "en": "Disagreements"},
    "telemetry_rate": {"tr": "Anlaşmazlık Oranı", "en": "Disagreement Rate"},
    "telemetry_top": {"tr": "En çok tekrarlayan anlaşmazlık kombinasyonları:", "en": "Most frequent disagreement combinations:"},
    "telemetry_empty": {"tr": "Henüz anlaşmazlık kaydı yok.", "en": "No disagreement records yet."},
    "telemetry_clear": {"tr": "Telemetri kayıtlarını temizle", "en": "Clear telemetry records"},
    "telemetry_cleared": {"tr": "{} kayıt silindi.", "en": "{} records deleted."},
    "icd10_label": {"tr": "ICD-10 Kodu:", "en": "ICD-10 Code:"},
    "icd10_chapter": {"tr": "ICD-10 Bölümü:", "en": "ICD-10 Chapter:"},
    "icd10_missing": {"tr": "ICD-10 kodu bulunamadı.", "en": "No ICD-10 code available."},
    "red_flag_title": {"tr": "🚨 Kırmızı Bayrak — Acil Tıbbi Değerlendirme Gerekli", "en": "🚨 Red Flag — Urgent Medical Assessment Needed"},
    "red_flag_none": {"tr": "Seçilen belirtiler için kırmızı bayrak kuralı tetiklenmedi.", "en": "No red-flag rule triggered by the selected symptoms."},
    "red_flag_caption": {
        "tr": "Bu uyarılar karar desteği amaçlıdır; kesin tanı koymaz. Şiddetli belirtilerde daima bir sağlık profesyoneline danışın.",
        "en": "These warnings are decision-support only. Always consult a healthcare professional for severe symptoms.",
    },
    "differential_title": {"tr": "🩺 Ayırıcı Tanı Listesi (Komorbidite / Çoklu Tanı)", "en": "🩺 Differential Diagnosis (Comorbidity / Multiple Conditions)"},
    "differential_caption": {
        "tr": "Tüm 7 modelin doğruluk ağırlıklı oylarıyla sıralanan en olası tanılar; hasta birden fazla durumu aynı anda taşıyabilir.",
        "en": "Top candidates ranked by the accuracy-weighted vote of all 7 models; a patient can carry more than one condition at once.",
    },
    "differential_disease": {"tr": "Olası Tanı", "en": "Candidate"},
    "differential_score": {"tr": "Ağırlıklı Skor (%)", "en": "Weighted Score (%)"},
    "differential_support": {"tr": "Destekleyen Model", "en": "Supporting Models"},
    "differential_support_count": {"tr": "Destek Sayısı", "en": "Support Count"},
    "differential_note": {
        "tr": "ℹ️ Aynı anda birden fazla hastalık olabilir; ayırıcı tanı listesi tek bir sonuçtan daha güvenilir bir klinik resim sunar.",
        "en": "ℹ️ Multiple conditions may coexist; the differential list offers a more reliable clinical picture than a single label.",
    },
}


def translate(language: str, key: str) -> str:
    """Return the translated string for a UI key (falls back to Turkish)."""

    entry = _TRANSLATIONS.get(key, {})
    if language in entry:
        return entry[language]
    return entry.get(DEFAULT_LANGUAGE, key)
