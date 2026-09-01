"""Shared utilities for the disease prediction project.

This module centralizes path handling, symptom normalization, and small helper
functions that are reused across preprocessing, model training, and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import re
import unicodedata

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_DELIMITER = ";"
TARGET_COLUMN = "prognosis"
MODEL_EXT = ".pkl"
DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "Training.csv": ("training_duzenlenmis.csv",),
    "Synthetic.csv": ("synthetic_dataset.csv",),
}


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_path(file_name: str) -> Path:
    """Return the first existing path for a dataset file.

    The project supports both the requested `data/` folder structure and the
    workspace's current flat layout, so the function checks multiple locations.
    """

    alias_names = DATASET_ALIASES.get(file_name, ())
    candidates = [
        DATA_DIR / file_name,
        *(DATA_DIR / alias_name for alias_name in alias_names),
        PROJECT_DIR / file_name,
        *(PROJECT_DIR / alias_name for alias_name in alias_names),
        PROJECT_DIR.parent / file_name,
        *(PROJECT_DIR.parent / alias_name for alias_name in alias_names),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if file_name == "Synthetic.csv":
        from src.synthetic_data import generate_and_persist_synthetic_dataset

        generate_and_persist_synthetic_dataset()
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Data file not found: {file_name}")


def resolve_model_path(file_name: str) -> Path:
    """Return the absolute path for a model artifact inside `models/`."""

    ensure_directory(MODELS_DIR)
    return MODELS_DIR / file_name


def load_dataset(file_name: str, delimiter: str = DEFAULT_DELIMITER) -> pd.DataFrame:
    """Load a semicolon-delimited dataset into a pandas DataFrame."""

    file_path = resolve_data_path(file_name)
    # Try common encodings when reading potentially non-UTF-8 CSV files
    encodings_to_try = ("utf-8", "cp1254", "latin1")
    for enc in encodings_to_try:
        try:
            return pd.read_csv(file_path, sep=delimiter, encoding=enc)
        except UnicodeDecodeError:
            # try the next encoding
            continue
    # As a last resort, read with replacement of invalid bytes so the app can still run
    return pd.read_csv(file_path, sep=delimiter, encoding="utf-8", errors="replace")


def normalize_symptom_name(value: str) -> str:
    """Normalize symptom labels so lookups are stable across sources."""

    cleaned = value.strip().lower()
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.replace("-", "_")
    cleaned = cleaned.replace("__", "_")
    return cleaned


TURKISH_SYMPTOM_ALIAS_MAP: dict[str, str] = {
    "kasinti": "itching",
    "cilt_dokuntusu": "skin_rash",
    "dugum_noktalarinda_deri_dokuntuleri": "nodal_skin_eruptions",
    "surekli hapsirma": "continuous_sneezing",
    "surekli_hapsirma": "continuous_sneezing",
    "titreme": "shivering",
    "eklem_agrisi": "joint_pain",
    "mide_agrisi": "stomach_pain",
    "asitlik": "acidity",
    "dildeki_ulserler": "ulcers_on_tongue",
    "kas_kaybi": "muscle_wasting",
    "kusma": "vomiting",
    "idrar_yaparken_yanma_hissi": "burning_micturition",
    "idrar_lekelenmesi": "spotting_urination",
    "tukenmislik": "fatigue",
    "kilo_alimi": "weight_gain",
    "endise": "anxiety",
    "soguk_eller_ve_ayaklar": "cold_hands_and_feets",
    "ruh_hali_degisiklikleri": "mood_swings",
    "kilo_verme": "weight_loss",
    "huzursuzluk": "restlessness",
    "letarji": "lethargy",
    "bogazdaki_yamalar": "patches_in_throat",
    "duzensiz_seker_seviyesi": "irregular_sugar_level",
    "oksuruk": "cough",
    "yuksek_ates": "high_fever",
    "cukur_gozler": "sunken_eyes",
    "nefes_darligi": "breathlessness",
    "terleme": "sweating",
    "dehidratasyon": "dehydration",
    "hazimsizlik": "indigestion",
    "bas_agrisi": "headache",
    "sarimsi_cilt": "yellowish_skin",
    "koyu_idrar": "dark_urine",
    "bulanti": "nausea",
    "istah_kaybi": "loss_of_appetite",
    "gozlerin_ardindaki_aci": "pain_behind_the_eyes",
    "sirt_agrisi": "back_pain",
    "kabizlik": "constipation",
    "karin_agrisi": "abdominal_pain",
    "ishal": "diarrhoea",
    "hafif_ates": "mild_fever",
    "sari_idrar": "yellow_urine",
    "gozlerin_sararmasi": "yellowing_of_eyes",
    "akut_karaciger_yetmezligi": "acute_liver_failure",
    "sivi_asiri_yuklenmesi": "fluid_overload",
    "mide_siskmesi": "swelling_of_stomach",
    "sismis_lenf_dugumleri": "swelled_lymph_nodes",
    "halsizlik": "malaise",
    "bulanık_ve_bozuk_gorus": "blurred_and_distorted_vision",
    "balgam": "phlegm",
    "bogaz_tahrişi": "throat_irritation",
    "gozlerde_kizariklik": "redness_of_eyes",
    "sinus_basinci": "sinus_pressure",
    "burun_akmasi": "runny_nose",
    "trafik_sikisikligi": "congestion",
    "gogus_agrisi": "chest_pain",
    "uzuvlarda_zayiflik": "weakness_in_limbs",
    "hizli_kalp_atisi": "fast_heart_rate",
    "bagirsak_hareketleri_sirasinda_aci": "pain_during_bowel_movements",
    "anal_bolgede_aci": "pain_in_anal_region",
    "kanli_disk": "bloody_stool",
    "anuste_tahriş": "irritation_in_anus",
    "boyun_agrisi": "neck_pain",
    "bas_donmesi": "dizziness",
    "kramplar": "cramps",
    "morarma": "bruising",
    "obezite": "obesity",
    "sismis_bacaklar": "swollen_legs",
    "sismis_kan_damarları": "swollen_blood_vessels",
    "sismis_yuz_ve_gozler": "puffy_face_and_eyes",
    "buyumus_tiroid": "enlarged_thyroid",
    "kirilgan_tirnaklar": "brittle_nails",
    "sismis_uzuvlar": "swollen_extremeties",
    "asiri_aclik": "excessive_hunger",
    "evlilik_disi_iliskiler": "extra_marital_contacts",
    "dudaklarda_kuruluk_ve_karincalanma": "drying_and_tingling_lips",
    "konusma_bozuklugu": "slurred_speech",
    "diz_agrisi": "knee_pain",
    "kalca_eklemi_agrisi": "hip_joint_pain",
    "kas_zayifligi": "muscle_weakness",
    "boyun_tutulmasi": "stiff_neck",
    "eklem_sislikleri": "swelling_joints",
    "hareket_sertligi": "movement_stiffness",
    "donme_hareketleri": "spinning_movements",
    "denge_kaybi": "loss_of_balance",
    "dengesizlik": "unsteadiness",
    "vucudun_bir_tarafinin_zayifligi": "weakness_of_one_body_side",
    "koku_kaybi": "loss_of_smell",
    "mesane_rahatsizligi": "bladder_discomfort",
    "idrarin_kotu_kokusu": "foul_smell_of_urine",
    "surekli_idrar_yapma_hissi": "continuous_feel_of_urine",
    "gazlarin_gecisi": "passage_of_gases",
    "ic_kasinti": "internal_itching",
    "zehirli_gorunum_tifo": "toxic_look_(typhos)",
    "depresyon": "depression",
    "sinirlilik": "irritability",
    "kas_agrisi": "muscle_pain",
    "degismis_duyusal_algi": "altered_sensorium",
    "vucutta_kirmizi_lekeler": "red_spots_over_body",
    "anormal_adet_kanamasi": "abnormal_menstruation",
    "diskromik_yamalar": "dischromic_patches",
    "gozlerden_su_akmasi": "watering_from_eyes",
    "istah_artisi": "increased_appetite",
    "poliuri": "polyuria",
    "aile_tarihi": "family_history",
    "mukoid_balgam": "mucoid_sputum",
    "pasli_balgam": "rusty_sputum",
    "konsantrasyon_eksikligi": "lack_of_concentration",
    "gorsel_bozukluklar": "visual_disturbances",
    "kan_transfuzyonu_almak": "receiving_blood_transfusion",
    "steril_olmayan_enjeksiyonlarin_alinmasi": "receiving_unsterile_injections",
    "koma": "coma",
    "mide_kanamasi": "stomach_bleeding",
    "karin_siskinligi": "distention_of_abdomen",
    "alkol_tuketiminin_tarihcesi": "history_of_alcohol_consumption",
    "balgamda_kan": "blood_in_sputum",
    "baldirda_belirgin_damarlar": "prominent_veins_on_calf",
    "carpinti": "palpitations",
    "acili_yuruyus": "painful_walking",
    "iltihapli_sivilceler": "pus_filled_pimples",
    "siyah_noktalar": "blackheads",
    "cirpinmak": "scurring",
    "cilt_soyma": "skin_peeling",
    "gumus_tozu_gibi": "silver_like_dusting",
    "tirnaklardaki_kucuk_cukurlar": "small_dents_in_nails",
    "iltihapli_tirnaklar": "inflammatory_nails",
}


SYMPTOM_TRANSLATIONS: dict[str, str] = {
    "itching": "Kaşıntı",
    "skin_rash": "Deri Döküntüsü",
    "nodal_skin_eruptions": "Nodüler Deri Döküntüleri",
    "continuous_sneezing": "Sürekli Hapşırma",
    "shivering": "Titreme",
    "chills": "Üşüme",
    "joint_pain": "Eklem Ağrısı",
    "stomach_pain": "Mide Ağrısı",
    "acidity": "Asitlik",
    "ulcers_on_tongue": "Dilde Ülserler",
    "muscle_wasting": "Kas Kaybı",
    "vomiting": "Kusma",
    "burning_micturition": "İdrarda Yanma",
    "spotting_urination": "İdrar Lekelenmesi",
    "fatigue": "Yorgunluk",
    "weight_gain": "Kilo Alımı",
    "anxiety": "Anksiyete",
    "cold_hands_and_feets": "Soğuk El ve Ayaklar",
    "mood_swings": "Duygu Durum Dalgalanmaları",
    "weight_loss": "Kilo Kaybı",
    "restlessness": "Huzursuzluk",
    "lethargy": "Letarji",
    "patches_in_throat": "Boğazda Plaklar",
    "irregular_sugar_level": "Düzensiz Şeker Düzeyi",
    "cough": "Öksürük",
    "high_fever": "Yüksek Ateş",
    "sunken_eyes": "Çökmüş Gözler",
    "breathlessness": "Nefes Darlığı",
    "sweating": "Terleme",
    "dehydration": "Dehidratasyon",
    "indigestion": "Hazımsızlık",
    "headache": "Baş Ağrısı",
    "yellowish_skin": "Sarımsı Cilt",
    "dark_urine": "Koyu İdrar",
    "nausea": "Bulantı",
    "loss_of_appetite": "İştahsızlık",
    "pain_behind_the_eyes": "Göz Arkası Ağrısı",
    "back_pain": "Sırt Ağrısı",
    "constipation": "Kabızlık",
    "abdominal_pain": "Karın Ağrısı",
    "diarrhoea": "İshal",
    "mild_fever": "Hafif Ateş",
    "yellow_urine": "Sarı İdrar",
    "yellowing_of_eyes": "Gözlerde Sararma",
    "acute_liver_failure": "Akut Karaciğer Yetmezliği",
    "fluid_overload": "Sıvı Yüklenmesi",
    "swelling_of_stomach": "Mide Şişkinliği",
    "swelled_lymph_nodes": "Şişmiş Lenf Düğümleri",
    "malaise": "Halsizlik",
    "blurred_and_distorted_vision": "Bulanık ve Bozuk Görüş",
    "phlegm": "Balgam",
    "throat_irritation": "Boğaz Tahrişi",
    "redness_of_eyes": "Göz Kızarıklığı",
    "sinus_pressure": "Sinüs Basıncı",
    "runny_nose": "Burun Akıntısı",
    "congestion": "Burun Tıkanıklığı",
    "chest_pain": "Göğüs Ağrısı",
    "weakness_in_limbs": "Uzuvlarda Güçsüzlük",
    "fast_heart_rate": "Hızlı Kalp Atışı",
    "pain_during_bowel_movements": "Dışkılama Sırasında Ağrı",
    "pain_in_anal_region": "Anal Bölgede Ağrı",
    "bloody_stool": "Kanlı Dışkı",
    "irritation_in_anus": "Anüste Tahriş",
    "neck_pain": "Boyun Ağrısı",
    "dizziness": "Baş Dönmesi",
    "cramps": "Kramp",
    "bruising": "Morarma",
    "obesity": "Obezite",
    "swollen_legs": "Şişmiş Bacaklar",
    "swollen_blood_vessels": "Şişmiş Kan Damarları",
    "puffy_face_and_eyes": "Şişkin Yüz ve Gözler",
    "enlarged_thyroid": "Tiroid Büyümesi",
    "brittle_nails": "Kırılgan Tırnaklar",
    "swollen_extremeties": "Şişmiş Uzuvlar",
    "excessive_hunger": "Aşırı Açlık",
    "extra_marital_contacts": "Evlilik Dışı Cinsel Temas",
    "drying_and_tingling_lips": "Dudaklarda Kuruluk ve Karıncalanma",
    "slurred_speech": "Konuşmada Pelteklik",
    "knee_pain": "Diz Ağrısı",
    "hip_joint_pain": "Kalça Eklem Ağrısı",
    "muscle_weakness": "Kas Güçsüzlüğü",
    "stiff_neck": "Boyun Tutulması",
    "swelling_joints": "Eklem Şişliği",
    "movement_stiffness": "Hareket Sertliği",
    "spinning_movements": "Dönme Hissi",
    "loss_of_balance": "Denge Kaybı",
    "unsteadiness": "Dengesizlik",
    "weakness_of_one_body_side": "Vücudun Bir Yanında Güçsüzlük",
    "loss_of_smell": "Koku Kaybı",
    "bladder_discomfort": "Mesane Rahatsızlığı",
    "foul_smell_of_urine": "İdrarda Kötü Koku",
    "continuous_feel_of_urine": "Sürekli İdrar Hissi",
    "passage_of_gases": "Gaz Çıkışı",
    "internal_itching": "İç Kaşıntı",
    "toxic_look_(typhos)": "Toksik Görünüm (Tifo)",
    "depression": "Depresyon",
    "irritability": "Sinirlilik",
    "muscle_pain": "Kas Ağrısı",
    "altered_sensorium": "Değişmiş Bilinç",
    "red_spots_over_body": "Vücutta Kırmızı Lekeler",
    "belly_pain": "Göbek Ağrısı",
    "abnormal_menstruation": "Anormal Adet Kanaması",
    "dischromic_patches": "Diskromik Lekeler",
    "watering_from_eyes": "Gözlerde Sulanma",
    "increased_appetite": "İştah Artışı",
    "polyuria": "Çok İdrara Çıkma",
    "family_history": "Aile Öyküsü",
    "mucoid_sputum": "Mukuslu Balgam",
    "rusty_sputum": "Pas Rengi Balgam",
    "lack_of_concentration": "Konsantrasyon Eksikliği",
    "visual_disturbances": "Görme Bozuklukları",
    "receiving_blood_transfusion": "Kan Nakli Almak",
    "receiving_unsterile_injections": "Steril Olmayan Enjeksiyon",
    "coma": "Koma",
    "stomach_bleeding": "Mide Kanaması",
    "distention_of_abdomen": "Karında Şişkinlik",
    "history_of_alcohol_consumption": "Alkol Kullanım Öyküsü",
    "blood_in_sputum": "Balgamda Kan",
    "prominent_veins_on_calf": "Baldırda Belirgin Damarlar",
    "palpitations": "Çarpıntı",
    "painful_walking": "Ağrılı Yürüme",
    "pus_filled_pimples": "İltihaplı Sivilceler",
    "blackheads": "Siyah Noktalar",
    "scurring": "Kaşınma",
    "skin_peeling": "Cilt Soyulması",
    "silver_like_dusting": "Gümüş Tozu Görünümü",
    "small_dents_in_nails": "Tırnaklarda Küçük Çukurlar",
    "inflammatory_nails": "İltihaplı Tırnaklar",
    "blister": "Su Kabarcığı",
    "red_sore_around_nose": "Burun Çevresinde Kırmızı Yara",
    "yellow_crust_ooze": "Sarı Kabuk ve Akıntı",
    "fluid_overload.1": "Sıvı Yüklenmesi",
}


def display_symptom_name(value: str, language: str = "tr") -> str:
    """Return the localized (Turkish or English) display name of a symptom."""

    if language == "tr":
        return SYMPTOM_TRANSLATIONS.get(value, humanize_label(value))
    return humanize_label(value)


def normalize_search_text(value: str) -> str:
    """Normalize text for user search by removing accents and punctuation."""

    if value is None:
        return ""

    cleaned = value.strip().lower()
    cleaned = cleaned.replace("ş", "s").replace("ç", "c").replace("ğ", "g").replace("ı", "i").replace("ö", "o").replace("ü", "u")
    cleaned = cleaned.replace("Ş", "s").replace("Ç", "c").replace("Ğ", "g").replace("İ", "i").replace("Ö", "o").replace("Ü", "u")
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def expand_search_terms(value: str) -> set[str]:
    """Generate normalized search terms including Turkish symptom aliases."""

    normalized = normalize_search_text(value)
    if not normalized:
        return {""}

    terms: set[str] = {normalized}
    terms.update(term for term in re.split(r"[_\s]+", normalized) if term)

    for alias, english_term in TURKISH_SYMPTOM_ALIAS_MAP.items():
        if alias == normalized or alias in normalized or normalized in alias:
            terms.add(english_term)
            terms.update(term for term in re.split(r"[_\s]+", english_term) if term)
        elif any(token in alias for token in terms):
            terms.add(english_term)
            terms.update(term for term in re.split(r"[_\s]+", english_term) if term)

    for english_term, turkish_name in SYMPTOM_TRANSLATIONS.items():
        turkish_norm = normalize_search_text(turkish_name)
        if turkish_norm == normalized or turkish_norm in normalized or normalized in turkish_norm:
            terms.add(english_term)
            terms.update(term for term in re.split(r"[_\s]+", english_term) if term)
        elif any(token in turkish_norm for token in terms):
            terms.add(english_term)
            terms.update(term for term in re.split(r"[_\s]+", english_term) if term)
    return terms


def humanize_label(value: str) -> str:
    """Convert a normalized symptom label into a display-friendly label."""

    return value.replace("_", " ").strip().title()


def save_model(model: object, file_name: str) -> Path:
    """Persist a trained estimator with joblib."""

    model_path = resolve_model_path(file_name)
    joblib.dump(model, model_path)
    return model_path


def load_model(file_name: str) -> object:
    """Load a persisted estimator from the models directory."""

    model_path = resolve_model_path(file_name)
    return joblib.load(model_path)


def deduplicate_preserve_order(values: Iterable[str]) -> list[str]:
    """Remove duplicates from an iterable while keeping the original order."""

    seen: set[str] = set()
    ordered_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered_values.append(value)
    return ordered_values


def ensure_columns_exist(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when one or more required columns are missing."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
