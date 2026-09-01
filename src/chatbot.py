"""Knowledge-driven Turkish chat assistant for the disease prediction app.

The assistant combines three data sources to produce useful answers:

* a knowledge base derived from the training data (which symptoms appear in
  which diseases, and how often),
* the live prediction bundle produced by the models for the current symptom
  selection,
* simple intent detection over the user's Turkish or English message.

All answers are decision-support only and never replace a medical opinion.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.disease_info import DISEASE_INFO
from src.red_flags import RED_FLAG_ADVICE, check_red_flags
from src.utils import (
    TARGET_COLUMN,
    TURKISH_SYMPTOM_ALIAS_MAP,
    humanize_label,
    normalize_search_text,
)

DISCLAIMER = (
    "Bu bilgiler istatistiksel karar desteği amaçlıdır; kesin tanı koymaz. "
    "Gerçek bir tıbbi değerlendirme için mutlaka bir sağlık profesyoneline danışın."
)

DISCLAIMER_EN = (
    "This information is statistical decision support only and does not make a diagnosis. "
    "Always consult a healthcare professional for a real medical evaluation."
)


# ---------------------------------------------------------------------------
# Content registry: the medical/linguistic dictionaries below live in
# data/content/*.json so that content updates never require code changes.
# Loading fails fast when a content file is missing or invalid.
# ---------------------------------------------------------------------------

CONTENT_DIR = Path(__file__).resolve().parent.parent / "data" / "content"


def _load_json_content(file_name: str) -> dict[str, Any]:
    """Load a content JSON file from data/content (fails fast when absent)."""

    path = CONTENT_DIR / file_name
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Medical content file missing: {path}. Restore data/content/*.json "
            "or re-run the app from the project root."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_str_dict(file_name: str) -> dict[str, str]:
    """Load a str -> str dictionary from JSON."""

    return {str(key): str(value) for key, value in _load_json_content(file_name).items()}


def _load_str_tuple_dict(file_name: str) -> dict[str, tuple[str, ...]]:
    """Load a str -> list[str] mapping from JSON, normalizing lists to tuples."""

    loaded = _load_json_content(file_name)
    return {str(key): tuple(str(item) for item in value) for key, value in loaded.items()}


SYMPTOM_ADVICE: dict[str, str] = _load_str_dict("symptom_advice.json")
SYMPTOM_DESCRIPTIONS: dict[str, str] = _load_str_dict("symptom_descriptions.json")
TURKISH_DISEASE_ALIASES: dict[str, tuple[str, ...]] = _load_str_tuple_dict("turkish_disease_aliases.json")
EXTRA_SYMPTOM_ALIASES: dict[str, tuple[str, ...]] = _load_str_tuple_dict("extra_symptom_aliases.json")

_TURKISH_SUFFIXES = (
    "yorsunuz",
    "yorum",
    "yorlar",
    "yoruz",
    "gumuz",
    "umuz",
    "iniz",
    "lerim",
    "lerin",
    "imiz",
    "mesi",
    "masi",
    "musu",
    "gum",
    "gin",
    "gim",
    "ugum",
    "ukum",
    "ugun",
    "eginiz",
    "yor",
    "sini",
    "den",
    "dan",
    "lar",
    "ler",
    "inde",
    "ina",
    "ini",
    "inin",
    "de",
    "da",
    "im",
    "in",
    "un",
    "um",
    "uz",
    "u",
    "i",
    "e",
    "a",
    "si",
    "yim",
    "sin",
    "niz",
    "miz",
)

_FUZZY_MIN_RATIO = 0.92


@dataclass(frozen=True)
class KnowledgeBase:
    """Derived statistics and match keys extracted from the training data."""

    symptom_columns: tuple[str, ...]
    diseases: tuple[str, ...]
    disease_symptoms: dict[str, tuple[tuple[str, float], ...]]
    symptom_diseases: dict[str, tuple[tuple[str, float], ...]]
    disease_symptom_freq: dict[str, dict[str, float]]
    symptom_keys: dict[str, frozenset[str]]
    disease_keys: dict[str, frozenset[str]]


def _normalize_disease_name(value: str) -> str:
    """Normalize a disease label for stable lookups."""

    return normalize_search_text(value)


def _build_match_keys(*names: str) -> frozenset[str]:
    """Build normalized, compact, and tokenized variants for text matching."""

    keys: set[str] = set()
    for name in names:
        normalized = normalize_search_text(name)
        if not normalized:
            continue
        keys.add(normalized)
        keys.add(normalized.replace("_", ""))
        keys.update(token for token in normalized.split("_") if len(token) >= 3)
    return frozenset(keys)


def build_knowledge_base(frame: pd.DataFrame, symptom_columns: list[str]) -> KnowledgeBase:
    """Derive per-disease and per-symptom statistics from the cleaned frame."""

    diseases = tuple(sorted(frame[TARGET_COLUMN].astype(str).unique()))
    disease_symptoms: dict[str, tuple[tuple[str, float], ...]] = {}
    disease_symptom_freq: dict[str, dict[str, float]] = {}
    symptom_diseases: dict[str, tuple[tuple[str, float], ...]] = {}

    for disease, group in frame.groupby(TARGET_COLUMN, sort=True):
        row_count = max(len(group), 1)
        counts = group[symptom_columns].sum()
        active = counts[counts > 0]
        sorted_items = sorted(active.items(), key=lambda item: item[1], reverse=True)
        disease_symptoms[str(disease)] = tuple(
            (symptom, round((count / row_count) * 100, 1)) for symptom, count in sorted_items
        )
        disease_symptom_freq[str(disease)] = {
            symptom: round((count / row_count) * 100, 1) for symptom, count in active.items()
        }

    english_to_turkish = {english: turkish for turkish, english in TURKISH_SYMPTOM_ALIAS_MAP.items()}
    for symptom in symptom_columns:
        group_stats = frame.groupby(TARGET_COLUMN)[symptom].mean()
        active = group_stats[group_stats > 0]
        symptom_diseases[symptom] = tuple(
            (str(disease), round((value * 100), 1)) for disease, value in active.items()
        )

    extended_aliases: dict[str, tuple[str, ...]] = {
        symptom: (alias,) for symptom, alias in english_to_turkish.items()
    }
    for symptom, extra_aliases in EXTRA_SYMPTOM_ALIASES.items():
        existing = extended_aliases.get(symptom)
        if existing:
            extended_aliases[symptom] = (*existing, *extra_aliases)
        else:
            extended_aliases[symptom] = extra_aliases

    symptom_keys = {
        symptom: _build_match_keys(
            symptom,
            humanize_label(symptom),
            *extended_aliases.get(symptom, ()),
        )
        for symptom in symptom_columns
    }
    normalized_disease_aliases = {
        normalize_search_text(disease_name): aliases
        for disease_name, aliases in TURKISH_DISEASE_ALIASES.items()
    }
    disease_keys = {
        disease: _build_match_keys(
            disease,
            humanize_label(disease),
            *normalized_disease_aliases.get(_normalize_disease_name(disease), ()),
        )
        for disease in diseases
    }

    return KnowledgeBase(
        symptom_columns=tuple(symptom_columns),
        diseases=diseases,
        disease_symptoms=disease_symptoms,
        symptom_diseases=symptom_diseases,
        disease_symptom_freq=disease_symptom_freq,
        symptom_keys=symptom_keys,
        disease_keys=disease_keys,
    )


def _strip_turkish_suffix(token: str) -> str:
    """Remove a common Turkish inflection suffix, keeping a meaningful stem."""

    for suffix in _TURKISH_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)]
    return token


def _candidate_forms(normalized_message: str) -> set[str]:
    """Return token, stemmed, and full-message forms used for matching."""

    forms: set[str] = {token for token in normalized_message.split("_") if len(token) >= 3}
    forms.add(normalized_message)
    stems = {_strip_turkish_suffix(token) for token in forms if len(token) >= 4}
    forms.update(stem for stem in stems if stem != normalized_message)
    return forms


_MATCH_MIN_SCORE = 4.0


def _contains_token_boundary(key: str, normalized_message: str) -> bool:
    """Return True when the key appears on word boundaries in the message."""

    start = normalized_message.find(key)
    while start != -1:
        before_ok = start == 0 or normalized_message[start - 1] == "_"
        after_index = start + len(key)
        after_ok = after_index == len(normalized_message) or normalized_message[after_index] == "_"
        if before_ok and after_ok:
            return True
        start = normalized_message.find(key, start + 1)
    return False


def _keys_match(keys: frozenset[str], normalized_message: str, forms: set[str]) -> float:
    """Return a weighted match score, or 0.0 when nothing meaningful matches."""

    score = 0.0
    seen: set[str] = set()

    def add_match(key: str, weight: float) -> None:
        nonlocal score
        if key in seen or len(key) < 3:
            return
        seen.add(key)
        score += len(key) * weight

    for key in keys:
        if len(key) >= 4 and _contains_token_boundary(key, normalized_message):
            add_match(key, 1.0)
        elif len(key) >= 8 and len(normalized_message) >= 8 and normalized_message in key:
            add_match(key, 0.8)

    for form in forms:
        for key in keys:
            if len(form) < 3:
                continue
            if key == form:
                add_match(key, 1.0)
            elif (
                len(form) >= 4
                and len(key) >= 5
                and len(key) >= len(form) + 2
                and key.startswith(form)
            ):
                add_match(key, 0.5)
            elif (
                len(key) >= 6
                and len(form) >= 6
                and difflib.SequenceMatcher(None, key, form).ratio() >= _FUZZY_MIN_RATIO
            ):
                add_match(key, 0.7)
    return score


def _detect_symptoms_scored(user_message: str, kb: KnowledgeBase) -> list[tuple[str, float]]:
    """Detect symptoms and return them sorted by match strength."""

    normalized = normalize_search_text(user_message)
    if not normalized:
        return []
    forms = _candidate_forms(normalized)
    matches: list[tuple[str, float]] = []
    for symptom, keys in kb.symptom_keys.items():
        score = _keys_match(keys, normalized, forms)
        if score >= _MATCH_MIN_SCORE:
            matches.append((symptom, score))
    return sorted(matches, key=lambda item: item[1], reverse=True)


def _detect_symptoms_llm(user_message: str, kb: KnowledgeBase, llm_settings: dict[str, Any] | None) -> list[str]:
    """Detect symptoms via the configured LLM, falling back to rule matching."""

    if llm_settings:
        from src.llm import extract_symptoms_with_llm

        llm_symptoms, _raw = extract_symptoms_with_llm(user_message, kb.symptom_columns, llm_settings)
        if llm_symptoms:
            return llm_symptoms
    return detect_symptoms(user_message, kb)


def _detect_diseases_scored(user_message: str, kb: KnowledgeBase) -> list[tuple[str, float]]:
    """Detect diseases and return them sorted by match strength."""

    normalized = normalize_search_text(user_message)
    if not normalized:
        return []
    forms = _candidate_forms(normalized)
    matches: list[tuple[str, float]] = []
    for disease, keys in kb.disease_keys.items():
        score = _keys_match(keys, normalized, forms)
        if score >= _MATCH_MIN_SCORE:
            matches.append((disease, score))
    return sorted(matches, key=lambda item: item[1], reverse=True)


def detect_symptoms(user_message: str, kb: KnowledgeBase) -> list[str]:
    """Find every symptom mentioned in the user's message."""

    return [symptom for symptom, _ in _detect_symptoms_scored(user_message, kb)[:5]]


def detect_diseases(user_message: str, kb: KnowledgeBase) -> list[str]:
    """Find every disease mentioned in the user's message."""

    return [disease for disease, _ in _detect_diseases_scored(user_message, kb)[:5]]


def format_selected_symptoms(selected_symptoms: list[str], limit: int = 5) -> str:
    """Return a short, human-friendly summary of the selected symptoms."""

    if not selected_symptoms:
        return "Henüz semptom seçilmedi."
    formatted = [humanize_label(symptom) for symptom in selected_symptoms]
    if len(formatted) <= limit:
        return ", ".join(formatted)
    remaining = len(formatted) - limit
    return f"{', '.join(formatted[:limit])} ve {remaining} tane daha"


def extract_question_topics(user_message: str) -> list[str]:
    """Extract a few searchable topics from the user's message."""

    cleaned = normalize_search_text(user_message)
    words = [word for word in cleaned.split("_") if len(word) >= 4]
    stop_words = {
        "bana",
        "bunu",
        "beni",
        "senin",
        "soru",
        "sorun",
        "boyle",
        "icin",
        "hangi",
        "neden",
        "nasil",
        "olan",
        "olur",
        "oldu",
        "lutfen",
        "var",
        "mı",
        "mi",
    }
    return [word for word in words if word not in stop_words][:4]


def _mention(marker: str, normalized: str) -> bool:
    """Return True when the marker appears as a token or substring."""

    return marker in normalized or any(token == marker for token in normalized.split("_"))


def _has_any(markers: tuple[str, ...], normalized: str) -> bool:
    """Return True when at least one marker appears inside the normalized text."""

    return any(marker in normalized for marker in markers)


def _clean_label(value: str) -> str:
    """Collapse duplicated whitespace inside a display label."""

    return " ".join(humanize_label(value).split())


def _format_freq(value: float) -> str:
    """Format a percentage value compactly."""

    if value >= 10 or value == 0:
        return f"%{value:.0f}"
    return f"%{value:.1f}"


def _selected_list(selected_symptoms: list[str]) -> str:
    """Describe the current symptom selection."""

    if not selected_symptoms:
        return ""
    return f" Şu an seçili listenizde: {format_selected_symptoms(selected_symptoms)}."


def _red_flag_warning(selected_symptoms: list[str]) -> str:
    """Return a warning paragraph when critical symptoms are selected."""

    warnings = [flag["advice_tr"] for flag in check_red_flags(selected_symptoms)]
    if not warnings:
        return ""
    return " Dikkat: " + " ".join(warnings)


def _symptom_explanation(
    symptom: str,
    all_symptoms: list[str],
    kb: KnowledgeBase,
    selected_symptoms: list[str],
    context: dict[str, Any],
) -> str:
    """Explain a symptom, the diseases where it is common, and sibling symptoms."""

    english_to_turkish = {english: turkish for turkish, english in TURKISH_SYMPTOM_ALIAS_MAP.items()}
    for symptom_name, extra_aliases in EXTRA_SYMPTOM_ALIASES.items():
        if symptom_name in english_to_turkish:
            continue
        english_to_turkish[symptom_name] = extra_aliases[0]
    turkish_name = english_to_turkish.get(symptom)
    title = humanize_label(symptom)
    if turkish_name:
        title = f"{turkish_name.replace('_', ' ').title()} ({title})"

    parts = [f"**{title}**"]
    description = SYMPTOM_DESCRIPTIONS.get(symptom)
    if description:
        parts.append(description)

    context_disease = context.get("disease")
    if context_disease and symptom in kb.disease_symptom_freq.get(context_disease, {}):
        freq = kb.disease_symptom_freq[context_disease][symptom]
        parts.append(
            f"Seçtiğin bağlam olan **{_clean_label(context_disease)}** kayıtlarının "
            f"{_format_freq(freq)}'inde bu belirti görülüyor."
        )

    top_diseases = kb.symptom_diseases.get(symptom, ())[:5]
    if top_diseases:
        disease_lines = [
            f"- {_clean_label(disease)} ({_format_freq(freq)} sıklık)" for disease, freq in top_diseases
        ]
        parts.append("Veri setinde bu belirtinin en sık eşleştiği hastalıklar:")
        parts.append("\n".join(disease_lines))

    other_symptoms = [item for item in all_symptoms if item != symptom]
    if other_symptoms:
        sibling_lines = []
        for other_symptom in other_symptoms[:3]:
            top_two = kb.symptom_diseases.get(other_symptom, ())[:2]
            if top_two:
                siblings = ", ".join(
                    f"{_clean_label(disease)} ({_format_freq(freq)})" for disease, freq in top_two
                )
            else:
                siblings = "belirti verisi bulunamadı"
            sibling_lines.append(f"- **{humanize_label(other_symptom)}**: en sık {siblings}")
        if sibling_lines:
            parts.append("Ayrıca şu belirtileri de fark ettim:\n" + "\n".join(sibling_lines))

    if symptom in selected_symptoms:
        parts.append("✅ Bu belirti şu an seçili listenizde bulunuyor.")
    else:
        parts.append("💡 Bu belirtiyi soldaki menüden seçip Analyze düğmesine basarak model sonucunu görebilirsiniz.")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


_DISEASE_SECTION_LABELS: dict[str, dict[str, str]] = {
    "tr": {
        "symptoms": "Belirtiler",
        "causes": "Nedenler",
        "risk_factors": "Risk faktörleri",
        "complications": "Olası komplikasyonlar",
        "when_to_see_doctor": "Ne zaman doktora başvurmalı",
        "diagnosis": "Tanı",
        "treatment": "Tedavi",
        "prevention": "Korunma",
    },
    "en": {
        "symptoms": "Symptoms",
        "causes": "Causes",
        "risk_factors": "Risk factors",
        "complications": "Possible complications",
        "when_to_see_doctor": "When to see a doctor",
        "diagnosis": "Diagnosis",
        "treatment": "Treatment",
        "prevention": "Prevention",
    },
}

_DISEASE_INFO_KEYS = (
    "description",
    "symptoms",
    "causes",
    "risk_factors",
    "complications",
    "when_to_see_doctor",
    "diagnosis",
    "treatment",
    "prevention",
)


def _disease_explanation(
    disease: str,
    kb: KnowledgeBase,
    selected_symptoms: list[str],
    final_prediction: str | None,
    language: str = "tr",
) -> str:
    """Explain a disease using curated medical info and dataset statistics."""

    lang = language if language in ("tr", "en") else "tr"
    labels = _DISEASE_SECTION_LABELS[lang]
    disclaimer = DISCLAIMER_EN if lang == "en" else DISCLAIMER
    info = DISEASE_INFO.get(disease, {})
    entry = info.get(lang) or info.get("tr") or {}

    parts = []
    if lang == "en":
        parts.append(f"**{_clean_label(disease)}**")
    else:
        parts.append(f"**{_clean_label(disease)}** hakkında")

    description = entry.get("description")
    if description:
        parts.append(description)

    dataset_lines = []
    top_symptoms = kb.disease_symptoms.get(disease, ())[:7]
    if top_symptoms and not entry:
        symptom_lines = [
            f"- {humanize_label(symptom)} ({_format_freq(freq)} sıklık)" for symptom, freq in top_symptoms
        ]
        if lang == "en":
            dataset_lines.append("Symptoms most frequently recorded with this disease in the dataset:")
        else:
            dataset_lines.append("Veri setinde bu hastalıkla birlikte en sık kaydedilen belirtiler:")
        dataset_lines.append("\n".join(symptom_lines))

    overlap = [s for s in selected_symptoms if s in dict(top_symptoms)]
    if overlap:
        overlap_text = ", ".join(humanize_label(s) for s in overlap)
        if lang == "en":
            dataset_lines.append(
                f"From your selected symptoms, these appear in this disease's typical list: {overlap_text}."
            )
        else:
            dataset_lines.append(
                f"Seçtiğiniz semptomlardan şunlar bu hastalığın tipik listesinde yer alıyor: {overlap_text}."
            )
    if final_prediction is not None and final_prediction == disease:
        if lang == "en":
            dataset_lines.append(
                "✅ The model's combined result also points to this disease, consistent with your symptoms."
            )
        else:
            dataset_lines.append(
                "✅ Modelin birleşik sonucu da bu hastalığı gösteriyor; seçtiğiniz belirtilerle uyumlu."
            )
    if dataset_lines:
        parts.append("\n".join(dataset_lines))

    for key in _DISEASE_INFO_KEYS:
        if key == "description":
            continue
        values = entry.get(key)
        if not values:
            continue
        if isinstance(values, str):
            parts.append(f"**{labels[key]}**\n{values}")
        else:
            bullet_lines = [f"- {value}" for value in values]
            parts.append(f"**{labels[key]}**\n" + "\n".join(bullet_lines))

    red_flags = [advice for symptom, advice in RED_FLAG_ADVICE.items() if symptom in selected_symptoms]
    if red_flags:
        parts.append("⚠️ " + " ".join(red_flags))
    parts.append(disclaimer)
    return "\n\n".join(parts)


def _next_steps_answer(selected_symptoms: list[str], final_prediction: str | None) -> str:
    """Suggest general and symptom-specific next steps."""

    advice_parts = [
        "Genel öneriler:\n- Dinlenme ve yeterli sıvı alımı\n- Belirtileri gün içinde düzenli takip et\n- Yeni ya da kötüleşen belirtilerde bir sağlık profesyoneline danış"
    ]
    specific = [SYMPTOM_ADVICE[symptom] for symptom in selected_symptoms if symptom in SYMPTOM_ADVICE]
    if specific:
        advice_parts.append("Seçtiğiniz belirtilere özel:\n- " + "\n- ".join(specific[:5]))

    warning = _red_flag_warning(selected_symptoms)
    if warning:
        advice_parts.append("⚠️" + warning)

    advice_parts.append(_selected_list(selected_symptoms) if selected_symptoms else "")
    advice_parts.append(DISCLAIMER)
    return "\n\n".join(part for part in advice_parts if part)


def _result_explanation(
    final_prediction: str,
    prediction_bundle: Any,
    selected_symptoms: list[str],
    model_accuracies: dict[str, float] | None,
    language: str = "tr",
) -> str:
    """Explain the consensus prediction in a natural, conversational tone."""

    lang = language if language in ("tr", "en") else "tr"
    disclaimer = DISCLAIMER_EN if lang == "en" else DISCLAIMER

    predictions = [
        prediction_bundle.decision_tree_prediction,
        prediction_bundle.naive_bayes_prediction,
        prediction_bundle.random_forest_prediction,
        prediction_bundle.logistic_regression_prediction,
        prediction_bundle.svm_prediction,
    ]
    unique_predictions = len(set(predictions))
    if lang == "en":
        if unique_predictions == 1:
            agreement = "All five models agree; this is a strong sign of consistency."
        elif unique_predictions == 2:
            agreement = "Most models agree; the agreement is moderate to good."
        elif unique_predictions >= 4:
            agreement = "The models disagree considerably, which points to real uncertainty."
        else:
            agreement = "The models differ; this indicates some uncertainty."
    else:
        if unique_predictions == 1:
            agreement = "Beş model de aynı sonucu verdi; bu güçlü bir uyum işaretidir."
        elif unique_predictions == 2:
            agreement = "Modellerin çoğunluğu aynı sonucu verdi; uyum orta-iyi düzeyde."
        elif unique_predictions >= 4:
            agreement = "Modeller oldukça farklı sonuçlar verdi; bu ciddi belirsizliğe işaret eder."
        else:
            agreement = "Modeller farklı sonuçlar verdi; bu durum belirsizliğe işaret eder."

    parts = [
        f"**{'Combined result: ' if lang == 'en' else 'Birleşik sonuç: '}{_clean_label(final_prediction)}**",
        agreement,
    ]

    if not prediction_bundle.apriori_rules.empty:
        top_rule = prediction_bundle.apriori_rules.iloc[0]
        if lang == "en":
            parts.append(
                f"Your symptoms most often co-occur with **{_clean_label(top_rule['consequent'])}** "
                f"in the dataset, which supports this result."
            )
        else:
            parts.append(
                f"Seçtiğiniz belirtiler veri setinde en çok **{_clean_label(top_rule['consequent'])}** "
                f"ile birlikte görülüyor; bu da sonucu destekleyen bir işaret."
            )

    if selected_symptoms:
        if lang == "en":
            parts.append(
                "Overview of your symptoms: " + format_selected_symptoms(selected_symptoms) + ". "
                "This is only a statistical assessment; please consult a doctor for a real diagnosis."
            )
        else:
            parts.append(
                "Belirtileriniz üzerinden özet: " + format_selected_symptoms(selected_symptoms) + ". "
                "Bu sonuç yalnızca istatistiksel bir değerlendirmedir; kesin tanı için bir hekime danışın."
            )

    parts.append(disclaimer)
    return "\n\n".join(part for part in parts if part)


def _differential_evaluation(
    final_prediction: str,
    prediction_bundle: Any,
    selected_symptoms: list[str],
    language: str = "tr",
) -> str:
    """Give a natural-language evaluation of the differential diagnosis list."""

    lang = language if language in ("tr", "en") else "tr"
    disclaimer = DISCLAIMER_EN if lang == "en" else DISCLAIMER

    differential = getattr(prediction_bundle, "differential_diagnosis", None) or []
    if not differential:
        if lang == "en":
            return (
                "No evaluation list is ready yet; please run the analysis with the **Analyze** "
                "button first.\n\n" + disclaimer
            )
        return (
            "Şu an değerlendirme listesi hazır değil; önce **Analyze** düğmesiyle analiz yapmanız "
            "gerekiyor.\n\n" + disclaimer
        )

    if lang == "en":
        parts = ["**Evaluation based on your symptoms**\n"]
        lines = []
        for entry in differential[:3]:
            disease = entry.get("disease", "")
            score = entry.get("score_pct", 0.0)
            info = DISEASE_INFO.get(disease, {}).get("en", {})
            description = info.get("description", "")
            detail = description.split(".")[0] if description else ""
            lines.append(f"- **{_clean_label(disease)}** ({score:.1f}% probability): {detail}")
        if lines:
            parts.append("\n".join(lines))
            parts.append(
                "This list ranks the most likely diseases in order; it does not make a diagnosis. "
                "If your symptoms are severe or getting worse, seek medical care."
            )
        else:
            parts.append("Not enough data; try selecting more symptoms.")
    else:
        parts = [f"**Seçtiğiniz belirtilere göre değerlendirme**\n"]
        lines = []
        for entry in differential[:3]:
            disease = entry.get("disease", "")
            score = entry.get("score_pct", 0.0)
            info = DISEASE_INFO.get(disease, {}).get("tr", {})
            description = info.get("description", "")
            detail = description.split(".")[0] if description else ""
            lines.append(f"- **{_clean_label(disease)}** (%{score:.1f} olasılık): {detail}")
        if lines:
            parts.append("\n".join(lines))
            parts.append(
                "Bu liste öncelik sırasına göre olası hastalıkları gösterir; kesin tanı koymaz. "
                "Belirtileriniz şiddetliyse veya kötüleşiyorsa mutlaka bir sağlık kuruluşuna başvurun."
            )
        else:
            parts.append("Yeterli veri bulunamadı; daha fazla belirti seçmeyi deneyin.")

    parts.append(disclaimer)
    return "\n\n".join(parts)


def _apriori_explanation(prediction_bundle: Any, language: str = "tr") -> str:
    """Explain the strongest Apriori rule in plain, conversational language."""

    lang = language if language in ("tr", "en") else "tr"
    disclaimer = DISCLAIMER_EN if lang == "en" else DISCLAIMER

    if prediction_bundle.apriori_rules.empty:
        if lang == "en":
            return (
                "No strong Apriori rule was found for this symptom combination. "
                "Selecting more symptoms increases the chance of a match.\n\n" + disclaimer
            )
        return (
            "Bu semptom kombinasyonu için güçlü bir Apriori kuralı bulunamadı. "
            "Daha fazla semptom seçerseniz eşleşme şansı artar.\n\n" + disclaimer
        )
    top_rule = prediction_bundle.apriori_rules.iloc[0]
    if lang == "en":
        return (
            f"The disease most frequently seen with your symptoms: **{_clean_label(top_rule['consequent'])}**.\n\n"
            f"In {top_rule['confidence_pct']}% of similar records these symptoms were recorded together with "
            f"this disease; a strong association. The rule shows a statistical association only, not a causal link.\n\n"
            + disclaimer
        )
    return (
        f"Seçtiğiniz belirtilerle en sık birlikte görülen hastalık: **{_clean_label(top_rule['consequent'])}**.\n\n"
        f"Veri setindeki benzer kayıtların %{top_rule['confidence_pct']}'inde bu belirtiler bu hastalıkla birlikte "
        f"kaydedilmiş; bu güçlü bir birliktelik işareti. Kural, seçilen semptomlar ile hastalık arasındaki "
        f"istatistiksel birlikteliği gösterir; neden-sonuç ilişkisi kurmaz.\n\n"
        + disclaimer
    )


def _probability_explanation(prediction_bundle: Any, final_prediction: str) -> str:
    """Explain the top Naive Bayes probability scores."""

    probability_frame = prediction_bundle.naive_bayes_probabilities
    if probability_frame is None or probability_frame.empty:
        return "Bu seçim için olasılık tablosu bulunamadı."
    top_rows = probability_frame.head(3)
    lines = [
        f"- {_clean_label(row['prognosis'])} → %{row['probability_pct']:.1f}"
        for _, row in top_rows.iterrows()
    ]
    return (
        f"Naive Bayes modelinin olasılık sıralaması:\n\n" + "\n".join(lines) + "\n\n"
        f"En yüksek olasılıklı sınıf birleşik sonuçla ({_clean_label(final_prediction)}) uyumlu olmayabilir; "
        "kesin sonuç için tüm model çıktıları birlikte değerlendirilmelidir.\n\n"
        + DISCLAIMER
    )


def _how_it_works_answer(model_accuracies: dict[str, float] | None) -> str:
    """Describe how the prediction system works in plain language."""

    return (
        "Sistem aslında basit bir mantıkla çalışıyor: soldan belirtilerinizi seçip **Analyze** "
        "düğmesine bastığınızda, aynı belirtilerin veri setinde hangi hastalıklarla birlikte "
        "görüldüğünü istatistiksel olarak tarar.\n\n"
        "1. **Apriori** — seçtiğiniz belirtilerle en sık birlikte görülen hastalıkları bulur.\n"
        "2. **7 sınıflandırma modeli** (Decision Tree, Naive Bayes, Random Forest, Logistic "
        "Regression, SVM, XGBoost ve LightGBM) her biri ayrı bir tahmin yapar.\n"
        "3. Tüm modellerin tahminleri birleştirilir ve **çoğunluk oyu** ile en olası hastalık "
        "belirlenir.\n\n"
        "Sonuç bir tanı değildir; sadece belirtilerinize dayanan istatistiksel bir ön "
        "değerlendirmedir. Kesin bilgi için mutlaka bir hekime danışın.\n\n"
        + DISCLAIMER
    )


def _greeting_answer() -> str:
    """Return the assistant's opening message."""

    return (
        "Merhaba! 👋 Ben bu arayüzün sohbet asistanıyım.\n\n"
        "Bana şunları sorabilirsin:\n"
        "- **Sonucu açıkla** — model sonuçlarını yorumlarım\n"
        "- **Apriori kuralını yorumla** — en güçlü birliktelik kuralını anlatırım\n"
        "- **Ne yapmalıyım?** — seçili belirtilere göre öneri veririm\n"
        "- **Hastalıklarımı değerlendir** — belirtilerinize göre olası hastalıkları sıralarım\n"
        "- **Dengue nedir?** / **Ateş hangi hastalıklarda görülür?** — hastalık ve belirti bilgisi veririm\n\n"
        "Belirtilerinizi yazarak da deneyebilirsiniz: *\"ateşim ve öksürüğüm var\"*."
    )


def _help_answer() -> str:
    """List the assistant's capabilities."""

    return (
        "Size yardımcı olabileceğim konular:\n\n"
        "🩺 **Sonuç yorumu** — \"Sonucu açıkla\" veya \"bu ne anlama geliyor?\"\n"
        "📊 **Apriori kuralı** — \"Apriori kuralını yorumla\"\n"
        "📝 **Sonraki adımlar** — \"Ne yapmalıyım?\" veya \"öneri ver\"\n"
        "🦠 **Hastalık bilgisi** — \"Dengue nedir?\", \"Zatürre belirtileri nelerdir?\"\n"
        "🤒 **Belirti bilgisi** — \"Ateş ne demek?\", \"Öksürük hangi hastalıklarda görülür?\"\n"
        "🔬 **Model bilgisi** — \"Modeller ne kadar doğru?\", \"Sistem nasıl çalışıyor?\"\n\n"
        "Ayrıca belirtilerinizi doğrudan yazabilirsiniz: *\"başım ağrıyor ve halsizim\"*."
    )


def build_chatbot_response(
    user_message: str,
    selected_symptoms: list[str],
    final_prediction: str | None,
    prediction_bundle: Any | None,
    context: dict[str, Any],
    kb: KnowledgeBase,
    model_accuracies: dict[str, float] | None = None,
    language: str = "tr",
    llm_settings: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate a contextual reply and the updated conversation context."""

    from src.i18n import translate
    from src.llm import build_llm_context, generate_llm_reply, llm_available

    if llm_settings and llm_available(llm_settings):
        context_text = build_llm_context(
            selected_symptoms,
            final_prediction,
            prediction_bundle,
            model_accuracies,
        )
        llm_reply = generate_llm_reply(user_message, context_text, llm_settings, language)
        if llm_reply:
            disclaimer = translate(language, "chat_caption")
            return f"{llm_reply}\n\n_{disclaimer}_", dict(context)

    normalized = normalize_search_text(user_message)
    stripped = user_message.strip()
    new_context = dict(context)
    tokens = normalized.split("_")
    short_message = len(tokens) <= 4

    if not normalized:
        return "Bir mesaj yazarsanız size yardımcı olabilirim. Örneğin: \"Sonucu açıkla\".", new_context

    scored_diseases = _detect_diseases_scored(user_message, kb)
    detected_diseases = [disease for disease, _ in scored_diseases]
    detected_symptoms = _detect_symptoms_llm(user_message, kb, llm_settings)
    disease_score = scored_diseases[0][1] if scored_diseases else 0.0

    if not detected_diseases and not detected_symptoms and short_message:
        if any(token in ("merhaba", "selam", "slm", "gunaydin", "iyi_gunler", "hello", "hi", "hey") for token in tokens):
            return _greeting_answer(), new_context
        if any(token in ("tesekkur", "tesekkurler", "sagol", "sagolun", "thanks", "thank") for token in tokens):
            return "Rica ederim! Başka bir sorunuz olursa buradayım. 😊", new_context
        if any(token in ("görüşürüz", "gorusuruz", "hoscakal", "bye", "gule_gule") for token in tokens):
            return "Görüşmek üzere, geçmiş olsun! 👋", new_context

    if _has_any(("yardim", "yapabilirsin", "yapabiliyorsun", "ne_yapabilirsin", "neler_yapabilirsin", "kullanilir", "nasil_kullanilir", "komut"), normalized):
        return _help_answer(), new_context

    if _has_any(("temizle", "sifirla", "yeniden_basla", "bastan_al"), normalized):
        return (
            "Sohbeti baştan başlatmak için paneldeki **🔄 Sohbeti Temizle** düğmesine basabilirsiniz. "
            "Semptom seçimini sıfırlamak için sol menüdeki **Reset** düğmesini kullanın.",
            new_context,
        )

    if _has_any(("dogruluk", "dogru", "accuracy", "guvenilir", "isabet", "ne_kadar"), normalized):
        if model_accuracies:
            accuracy_lines = [f"- {model}: %{round(score * 100, 1)}" for model, score in model_accuracies.items()]
            return (
                "Model doğrulukları (80/20 ayrımın doğrulama kısmında değerlendirme):\n\n"
                + "\n".join(accuracy_lines)
                + "\n\n"
                + "Bu değerler modelin görmediği doğrulama verisindeki performansı gösterir; gerçek dünyada daima bir hekim değerlendirmesi gerekir.\n\n"
                + DISCLAIMER,
                new_context,
            )
        return "Model doğrulukları şu anda yüklenemedi.", new_context

    if _has_any(("nasil_calisir", "nasil_calisiyor", "calisiyor", "calisir", "how_does_it_work", "how_does_the_system", "system_work"), normalized):
        return _how_it_works_answer(model_accuracies), new_context

    if prediction_bundle is not None and final_prediction is not None:
        if _has_any(("olasilik", "probability"), normalized):
            return _probability_explanation(prediction_bundle, final_prediction), new_context
        if _has_any(("degerlendir", "degerlendirme", "analiz", "evaluate"), normalized):
            return _differential_evaluation(final_prediction, prediction_bundle, selected_symptoms, language=language), new_context
        if _has_any(("sonuc", "tahmin", "diagnosis", "ne_anlama", "sonucu_acikla", "hastaligim", "ne_hastaligi", "hastaligini", "ne_olabilir", "explain_result", "result", "what_does_it_mean"), normalized):
            return _result_explanation(final_prediction, prediction_bundle, selected_symptoms, model_accuracies, language=language), new_context
        if _has_any(("apriori", "kural", "rule"), normalized):
            return _apriori_explanation(prediction_bundle, language=language), new_context

    if _has_any(("ne_yap", "yapmaliyim", "yapabilirim", "oneri", "tedavi", "tavsiye", "next", "help", "what_should_i_do", "what_to_do", "advice", "recommendation"), normalized):
        return _next_steps_answer(selected_symptoms, final_prediction), new_context

    if _has_any(("ciddi", "tehlikeli", "riskli", "kotu_mu"), normalized):
        disease = new_context.get("disease")
        disease_text = f"**{_clean_label(disease)}** hakkında" if disease else "Bu durum hakkında"
        warning = _red_flag_warning(selected_symptoms)
        return (
            f"{disease_text}: veri seti ciddiyet bilgisi içermiyor; belirtilerin şiddetini ve riski yalnızca bir hekim "
            "değerlendirebilir." + warning + "\n\n" + DISCLAIMER,
            new_context,
        )

    if detected_diseases:
        disease = detected_diseases[0]
        new_context["disease"] = disease
        if detected_symptoms and _has_any(("gorulur", "görülür", "var_mi", "varmi", "olur_mu", "olurmu", "gosterir", "gözlemlenir", "gozlemlenir", "yapiyor", "yapar"), normalized):
            full_name_matches = [
                disease_name
                for disease_name, _ in scored_diseases
                if _contains_token_boundary(_normalize_disease_name(disease_name), normalized)
            ]
            preferred_diseases = full_name_matches or detected_diseases
            frequency_lines = []
            for disease_name in preferred_diseases[:3]:
                for symptom in detected_symptoms[:3]:
                    freq = kb.disease_symptom_freq.get(disease_name, {}).get(symptom)
                    if freq is not None:
                        frequency_lines.append(
                            f"- **{humanize_label(symptom)}**: {_clean_label(disease_name)} kayıtlarının {_format_freq(freq)}'inde görülüyor."
                        )
            if frequency_lines:
                return (
                    "Veri setine göre:\n\n" + "\n".join(frequency_lines) + "\n\n" + DISCLAIMER,
                    new_context,
                )
        if _has_any(("nedir", "ne_demek", "hakkinda", "belirtileri", "belirtiler", "neden", "tanimla", "acikla", "neymis", "ne_hastaligi"), normalized) or (short_message and disease_score >= 5):
            return _disease_explanation(disease, kb, selected_symptoms, final_prediction, language=language), new_context

    if detected_symptoms:
        new_context["symptom"] = detected_symptoms[0]
        return _symptom_explanation(detected_symptoms[0], detected_symptoms, kb, selected_symptoms, new_context), new_context

    if selected_symptoms and not prediction_bundle and _has_any(("apriori", "kural"), normalized):
        return (
            "Henüz Apriori kuralları hesaplanmadı. Önce sol menüden belirtilerinizi seçip "
            "**Analyze** düğmesine basarsanız, seçtiğiniz belirtilerle en sık birlikte görülen "
            "hastalığı anlatırım.\n\n" + DISCLAIMER,
            new_context,
        )

    if selected_symptoms and not prediction_bundle and _has_any(("degerlendir", "analiz", "evaluate"), normalized):
        top_diseases = _top_diseases_for_symptoms(selected_symptoms, kb, top_n=5)
        if top_diseases:
            lines = [f"- {disease} ({_format_freq(freq)} sıklık)" for disease, freq in top_diseases]
            return (
                "Seçtiğiniz semptomlarla veri setinde en çok eşleşen hastalıklar:\n\n"
                + "\n".join(lines)
                + "\n\n"
                + "Kesin model çıktısı için **Analyze** düğmesine basmanız yeterli.\n\n"
                + DISCLAIMER,
                new_context,
            )

    if _has_any(("belirti", "semptom", "sikayet", "symptom"), normalized):
        return (
            "Seçili belirtileriniz: " + format_selected_symptoms(selected_symptoms) + ".\n\n"
            "Bu liste hakkında \"Sonucu açıkla\", \"Apriori kuralını yorumla\" veya \"Ne yapmalıyım?\" diyebilirsiniz.",
            new_context,
        )

    topics = extract_question_topics(user_message)
    if topics:
        return (
            f"'{stripped}' sorusunu doğrudan yanıtlamak için daha fazla bağlam gerekebilir; "
            f"{', '.join(topics)} konularında genel yorum yapabilirim.\n\n"
            "Deneyebileceğiniz sorular:\n- \"Sonucu açıkla\"\n- \"Ne yapmalıyım?\"\n- \"Ateş hangi hastalıklarda görülür?\"\n- \"Dengue nedir?\"",
            new_context,
        )

    return (
        "Bunu tam olarak anlayamadım. Şu konularda yardımcı olabilirim:\n\n"
        "- \"Sonucu açıkla\"\n- \"Apriori kuralını yorumla\"\n- \"Ne yapmalıyım?\"\n"
        "- \"Dengue nedir?\"\n- \"Ateş hangi hastalıklarda görülür?\"\n\n"
        "Ya da belirtilerinizi yazın: *\"başım ağrıyor ve halsizim\"*.",
        new_context,
    )


def _top_diseases_for_symptoms(
    selected_symptoms: list[str],
    kb: KnowledgeBase,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Rank diseases by how often they contain the selected symptoms."""

    if not selected_symptoms:
        return []
    scores: dict[str, float] = {}
    for symptom in selected_symptoms:
        for disease, freq in kb.symptom_diseases.get(symptom, ()):
            scores[disease] = scores.get(disease, 0.0) + freq
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def stream_chat_reply(text: str, word_delay: float = 0.02) -> Iterator[str]:
    """Yield a reply word-by-word so it can be streamed with st.write_stream."""

    for match in re.finditer(r"\S+\s*", text):
        yield match.group(0)
        time.sleep(word_delay)
    yield "\n"
