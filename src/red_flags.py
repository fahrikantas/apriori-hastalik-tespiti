"""Red-flag rules that warrant urgent medical attention.

Two rule families are supported:

* single-symptom flags (``RED_FLAG_ADVICE``) — any critical symptom on its own;
* combination flags (``RED_FLAG_COMBINATIONS``) — some symptom pairs/triples are
  only dangerous together (e.g. high fever + stiff neck strongly suggests
  meningitis) and are therefore checked as a set.

All advice is bilingual (Turkish/English) so the Streamlit UI, the REST API and
the chat assistant can consume the same rules. Decision-support only; it never
replaces a medical assessment.
"""

from __future__ import annotations

from typing import Any

# Single critical symptoms -> short Turkish advice (kept for chat/UI compatibility).
RED_FLAG_ADVICE: dict[str, str] = {
    "breathlessness": "Nefes darlığı ciddi bir belirtidir; acil tıbbi değerlendirme gerektirebilir.",
    "chest_pain": "Göğüs ağrısı kalp kaynaklı olabilir; vakit kaybetmeden sağlık kuruluşuna başvur.",
    "altered_sensorium": "Bilinç bulanıklığı acil durum belirtisidir; hemen tıbbi yardım al.",
    "coma": "Bilinç kaybı (koma) acil durumdur; hemen acil servise başvur.",
    "bloody_stool": "Kanlı dışkı ciddi bir durum olabilir; en kısa sürede doktora başvur.",
    "stomach_bleeding": "Mide kanaması şüphesi acil değerlendirme gerektirir.",
    "blood_in_sputum": "Balgamda kan görülmesi mutlaka incelenmelidir; doktora başvur.",
    "stiff_neck": "Boyun tutulması yüksek ateşle birlikteyse acil değerlendirme gerekir.",
    "weakness_of_one_body_side": "Vücudun bir tarafındaki güçsüzlük felç (inme) belirtisi olabilir; acilen acil servise başvur.",
    "slurred_speech": "Konuşma bozukluğu/geveleme felç (inme) belirtisi olabilir; hemen tıbbi yardım al.",
    "visual_disturbances": "Ani görme bozukluğu ciddi ve acil değerlendirme gerektirir.",
    "loss_of_balance": "Ani denge kaybı felç veya ciddi nörolojik durum belirtisi olabilir; acil değerlendirme gerekir.",
    "acute_liver_failure": "Akut karaciğer yetmezliği hayatı tehdit eder; derhal hastaneye başvur.",
    "dehydration": "Ciddi sıvı kaybı (dehidratasyon) acil sıvı tedavisi gerektirebilir.",
    "sunken_eyes": "Çukur gözler belirgin sıvı kaybına işaret edebilir; yakın takip ve değerlendirme gerekir.",
    "pain_behind_the_eyes": "Göz arkası ağrısı bazı ateşli hastalıklarda ağırlaşan bir durumun belirtisi olabilir; değerlendirme gerekir.",
    "foul_smell_of_urine": "Kötü kokulu idrar ciddi idrar yolu enfeksiyonu belirtisi olabilir.",
    "continuous_feel_of_urine": "Sürekli idrara çıkma hissi ciddi idrar yolu enfeksiyonu belirtisi olabilir.",
    "spotting_urination": "İdrar yaparken lekelenme/kanama ciddiye alınmalı; doktora başvur.",
}

# Symptom combinations that are only clinically dangerous when they occur together.
# 'symptoms' must all be present to trigger the rule.
RED_FLAG_COMBINATIONS: list[dict[str, Any]] = [
    {
        "id": "menengitis_suspicion",
        "symptoms": ["high_fever", "stiff_neck"],
        "severity": "critical",
        "advice": {
            "tr": "Yüksek ateş + boyun tutulması menenjit şüphesi taşır; vakit kaybetmeden acil servise başvur.",
            "en": "High fever with neck stiffness suggests meningitis; seek urgent medical attention immediately.",
        },
    },
    {
        "id": "menengitis_triple",
        "symptoms": ["high_fever", "stiff_neck", "vomiting"],
        "severity": "critical",
        "advice": {
            "tr": "Ateş, boyun tutulması ve kusma kümesi menenjit ile uyumlu olabilir; bu acil bir durumdur.",
            "en": "Fever, stiff neck and vomiting are consistent with meningitis; this is an emergency.",
        },
    },
    {
        "id": "cardiac_emergency",
        "symptoms": ["chest_pain", "breathlessness"],
        "severity": "critical",
        "advice": {
            "tr": "Göğüs ağrısı ile birlikte nefes darlığı kalp kaynaklı bir acil durum olabilir; acil servise başvur.",
            "en": "Chest pain with shortness of breath may be a cardiac emergency; seek emergency care.",
        },
    },
    {
        "id": "heart_attack_hint",
        "symptoms": ["chest_pain", "sweating", "nausea"],
        "severity": "critical",
        "advice": {
            "tr": "Göğüs ağrısı, terleme ve bulantı kalp krizi belirtisi olabilir; hemen acil servise başvur.",
            "en": "Chest pain with sweating and nausea may indicate a heart attack; call emergency services now.",
        },
    },
    {
        "id": "stroke_hint",
        "symptoms": ["weakness_of_one_body_side", "slurred_speech"],
        "severity": "critical",
        "advice": {
            "tr": "Vücudun bir tarafında güçsüzlük + konuşma bozukluğu felç (inme) belirtisidir. ZAMAN ÇOK DEĞERLİ — hemen acil servise başvur.",
            "en": "One-sided weakness with slurred speech are signs of stroke. TIME IS CRITICAL — call emergency services now.",
        },
    },
    {
        "id": "stroke_fall",
        "symptoms": ["weakness_of_one_body_side", "loss_of_balance", "dizziness"],
        "severity": "critical",
        "advice": {
            "tr": "Tek taraflı güçsüzlük, denge kaybı ve baş dönmesi felç (inme) belirtisi olabilir; acilen tıbbi değerlendirme gerekir.",
            "en": "One-sided weakness, loss of balance and dizziness may signal a stroke; urgent evaluation needed.",
        },
    },
    {
        "id": "brain_hemorrhage_hint",
        "symptoms": ["weakness_of_one_body_side", "headache", "altered_sensorium"],
        "severity": "critical",
        "advice": {
            "tr": "Felç benzeri belirtiler + ani baş ağrısı + bilinç değişikliği; beyin kanaması belirtisi olabilir. Acil durum.",
            "en": "Paralysis-like symptoms with sudden severe headache and altered consciousness can indicate a brain haemorrhage. Emergency.",
        },
    },
    {
        "id": "severe_dehydration",
        "symptoms": ["sunken_eyes", "dehydration", "diarrhoea"],
        "severity": "important",
        "advice": {
            "tr": "Çukur gözler + dehidratasyon + ishal ciddi sıvı kaybına işaret eder; hastada sıvı replasmanı gerekebilir.",
            "en": "Sunken eyes, dehydration and diarrhoea point to severe fluid loss; IV replacement may be needed.",
        },
    },
    {
        "id": "viral_bleeding_suspicion",
        "symptoms": ["pain_behind_the_eyes", "red_spots_over_body", "bruising"],
        "severity": "critical",
        "advice": {
            "tr": "Göz arkası ağrısı, vücutta mor/kırmızı lekeler ve kanama eğilimi hemorajik bir viral hastalık şüphesidir; acilen değerlendirilmelidir.",
            "en": "Eye socket pain, red spots on the body and bleeding tendency could be a haemorrhagic viral illness; urgently evaluated.",
        },
    },
    {
        "id": "sepsis_hint",
        "symptoms": ["high_fever", "chills", "altered_sensorium", "breathlessness"],
        "severity": "critical",
        "advice": {
            "tr": "Yüksek ateş, titreme ile birlikte bilinç ya da nefes değişikliği kan zehirlenmesi (sepsis) belirtisi olabilir; bu acil bir durumdur.",
            "en": "Fever with chills plus altered consciousness or breathlessness may indicate sepsis; this is an emergency.",
        },
    },
    {
        "id": "appendicitis_suspicion",
        "symptoms": ["abdominal_pain", "vomiting", "high_fever"],
        "severity": "critical",
        "advice": {
            "tr": "Karın ağrısı, kusma ve yüksek ateş apandisit şüphesi taşır; cerrahi değerlendirme gerekebilir.",
            "en": "Abdominal pain, vomiting and high fever raise appendicitis suspicion; surgical assessment may be required.",
        },
    },
]


def check_red_flags(selected_symptoms: list[str]) -> list[dict[str, Any]]:
    """Return every red-flag rule that matches the selected symptoms.

    Single-symptom rules are uppercase-only; each returned entry carries the
    ``bulgusu`` id, severity, the matched single symptoms and bilingual advice
    (keys ``advice_tr`` / ``advice_en``). Empty selection returns an empty list.
    """

    if not selected_symptoms:
        return []

    normalized = {str(symptom).strip().lower() for symptom in selected_symptoms}

    results: list[dict[str, Any]] = []

    for symptom, advice in RED_FLAG_ADVICE.items():
        if symptom in normalized:
            results.append(
                {
                    "id": f"single:{symptom}",
                    "rule": "single",
                    "severity": "critical",
                    "matched_symptoms": [symptom],
                    "advice_tr": advice,
                    "advice_en": _ADVICE_EN.get(symptom, advice),
                }
            )

    for combination in RED_FLAG_COMBINATIONS:
        combined = set(combination["symptoms"])
        if combined.issubset(normalized):
            results.append(
                {
                    "id": combination["id"],
                    "rule": "combination",
                    "severity": combination.get("severity", "critical"),
                    "matched_symptoms": list(combination["symptoms"]),
                    "advice_tr": combination["advice"]["tr"],
                    "advice_en": combination["advice"]["en"],
                }
            )

    return results


# English equivalents for the single-symptom advice (chatbot/API language switch).
_ADVICE_EN: dict[str, str] = {
    "breathlessness": "Shortness of breath is serious; urgent medical assessment may be required.",
    "chest_pain": "Chest pain can be cardiac; seek medical care without delay.",
    "altered_sensorium": "Altered consciousness is an emergency sign; get medical help now.",
    "coma": "Loss of consciousness (coma) is an emergency; call emergency services.",
    "bloody_stool": "Bloody stool may be serious; see a doctor as soon as possible.",
    "stomach_bleeding": "Possible stomach bleeding requires urgent assessment.",
    "blood_in_sputum": "Blood in sputum must be investigated; consult a doctor.",
    "stiff_neck": "Stiff neck with high fever requires urgent assessment.",
    "weakness_of_one_body_side": "One-sided limb weakness can be stroke; seek emergency care now.",
    "slurred_speech": "Slurred speech can be a sign of stroke; seek immediate help.",
    "visual_disturbances": "Sudden vision loss requires urgent assessment.",
    "loss_of_balance": "Sudden loss of balance may indicate stroke or a neurological condition; urgent assessment needed.",
    "acute_liver_failure": "Acute liver failure is life-threatening; go to the hospital urgently.",
    "dehydration": "Severe fluid loss (dehydration) may require urgent fluid therapy.",
    "sunken_eyes": "Sunken eyes indicate significant fluid loss; closely monitor.",
    "pain_behind_the_eyes": "Pain behind the eyes may signal a worsening fever illness; assessment needed.",
    "foul_smell_of_urine": "Foul-smelling urine could signal a serious urinary infection.",
    "continuous_feel_of_urine": "Continuous urge to urinate could signal a serious urinary infection.",
    "spotting_urination": "Blood-tinged urine should not be ignored; see a doctor.",
}