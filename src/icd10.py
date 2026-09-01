"""ICD-10 code mapping for every disease label used by the models.

Mapping the free-text disease names to standardized ICD-10 codes makes the
system interoperable with EHRs, insurance systems and other health tooling.
Codes are keyed by the exact prognosis labels present in the training data;
lookup is case-insensitive with a normalized fallback.

Decision-support context: codes are clinical approximations chosen for the
dataset's disease labels and must not be used for billing without clinical
verification.
"""

from __future__ import annotations

DISEASE_ICD10: dict[str, str] = {
    "(vertigo) Paroymsal  Positional Vertigo": "H81.1",
    "AIDS": "B24",
    "Acne": "L70.0",
    "Alcoholic hepatitis": "K70.1",
    "Allergy": "T78.40",
    "Arthritis": "M13.9",
    "Bronchial Asthma": "J45.9",
    "Chicken pox": "B01.9",
    "Chronic cholestasis": "K83.1",
    "Common Cold": "J00",
    "Dengue": "A90",
    "Diabetes": "E14.9",
    "Dimorphic hemmorhoids(piles)": "K64.9",
    "Drug Reaction": "T88.7",
    "Fungal infection": "B36.9",
    "GERD": "K21.9",
    "Gastroenteritis": "A09.9",
    "Heart attack": "I21.9",
    "Hepatitis B": "B16.9",
    "Hepatitis C": "B17.1",
    "Hepatitis D": "B17.0",
    "Hepatitis E": "B17.2",
    "Hypertension": "I10",
    "Hyperthyroidism": "E05.9",
    "Hypoglycemia": "E16.2",
    "Hypothyroidism": "E03.9",
    "Impetigo": "L01.0",
    "Jaundice": "R17",
    "Malaria": "B54",
    "Migraine": "G43.9",
    "Osteoarthristis": "M19.9",
    "Paralysis (brain hemorrhage)": "I61.9",
    "Peptic ulcer diseae": "K27.9",
    "Pneumonia": "J18.9",
    "Psoriasis": "L40.9",
    "Tuberculosis": "A15.9",
    "Typhoid": "A01.0",
    "Urinary tract infection": "N39.0",
    "Varicose veins": "I83.9",
    "hepatitis A": "B15.9",
}

# First letter of the code -> WHO ICD-10 chapter title (abbreviated).
ICD10_CHAPTERS: dict[str, str] = {
    "A": "Certain infectious and parasitic diseases",
    "B": "Certain infectious and parasitic diseases",
    "E": "Endocrine, nutritional and metabolic diseases",
    "G": "Diseases of the nervous system",
    "H": "Diseases of the eye and adnexa; ear and mastoid process",
    "I": "Diseases of the circulatory system",
    "J": "Diseases of the respiratory system",
    "K": "Diseases of the digestive system",
    "L": "Diseases of the skin and subcutaneous tissue",
    "M": "Diseases of the musculoskeletal system and connective tissue",
    "N": "Diseases of the genitourinary system",
    "R": "Symptoms, signs and abnormal clinical findings, not elsewhere classified",
    "T": "Injury, poisoning and certain other consequences of external causes",
}

_LOOKUP: dict[str, str] = {name.strip().lower(): code for name, code in DISEASE_ICD10.items()}


def get_icd10_code(disease_name: str | None) -> str | None:
    """Return the ICD-10 code for a disease label, or None when unknown."""

    if not disease_name:
        return None
    normalized = str(disease_name).strip().lower()
    code = _LOOKUP.get(normalized)
    if code is not None:
        return code
    # fall back to a fuzzy scan for labels that only differ in casing/spacing
    compressed = "".join(normalized.split())
    for canonical, canonical_code in _LOOKUP.items():
        if "".join(canonical.split()) == compressed:
            return canonical_code
    return None


def icd10_chapter(icd10_code: str | None) -> str | None:
    """Return the WHO chapter title for an ICD-10 code prefix."""

    if not icd10_code:
        return None
    return ICD10_CHAPTERS.get(str(icd10_code)[0].upper())


def icd10_summary(disease_name: str | None) -> dict[str, str | None]:
    """Return a compact ICD-10 summary for a disease label."""

    code = get_icd10_code(disease_name)
    if code is None:
        return {"code": None, "chapter": None}
    return {"code": code, "chapter": icd10_chapter(code)}


def all_disease_codes() -> dict[str, str]:
    """Return the complete disease -> ICD-10 mapping (stable copy)."""

    return dict(DISEASE_ICD10)
