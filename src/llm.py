"""Optional LLM integration (Ollama local / OpenAI / Anthropic) for the chat assistant."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterable

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT_TR = (
    "Sen semptom tabanlı hastalık tahmin uygulamasının Türkçe karar destek asistanısın. "
    "Kesin tanı koyma; her yanıtın sonunda tıbbi değerlendirme gerektiğini hatırlat. "
    "Verilen model sonuçlarını ve veri seti istatistiklerini yorumla."
)

SYSTEM_PROMPT_EN = (
    "You are the English decision-support assistant for a symptom-based disease prediction app. "
    "Never provide a definitive diagnosis; remind the user to consult a healthcare professional. "
    "Interpret the provided model outputs and dataset statistics."
)


def llm_available(settings: dict[str, Any]) -> bool:
    """Return True when an LLM backend is configured."""

    provider = settings.get("provider", "off")
    if provider == "ollama":
        return bool(settings.get("ollama_url") and settings.get("ollama_model"))
    if provider == "openai":
        return bool(settings.get("openai_api_key") and settings.get("openai_model"))
    if provider == "anthropic":
        return bool(
            (settings.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", ""))
            and settings.get("anthropic_model")
        )
    return False


def _build_messages(
    user_message: str,
    context_text: str,
    language: str,
) -> list[dict[str, str]]:
    system_prompt = SYSTEM_PROMPT_TR if language == "tr" else SYSTEM_PROMPT_EN
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Bağlam:\n{context_text}\n\nKullanıcı sorusu:\n{user_message}"
                if language == "tr"
                else f"Context:\n{context_text}\n\nUser question:\n{user_message}"
            ),
        },
    ]


def _call_ollama(messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
    url = settings["ollama_url"].rstrip("/") + "/api/chat"
    payload = {
        "model": settings["ollama_model"],
        "messages": messages,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("message", {}).get("content", "")).strip()


def _call_openai(messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
    api_key = settings.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": settings.get("openai_model", "gpt-4o-mini"),
        "messages": messages,
        "temperature": 0.3,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"]).strip()


def _call_anthropic(messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
    api_key = settings.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    payload = {
        "model": settings.get("anthropic_model", ANTHROPIC_MODEL),
        "max_tokens": 1024,
        "messages": [message for message in messages if message.get("role") != "system"],
        "system": next((message["content"] for message in messages if message.get("role") == "system"), ""),
    }
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return "".join(block.get("text", "") for block in body.get("content", [])).strip()


def generate_llm_reply(
    user_message: str,
    context_text: str,
    settings: dict[str, Any],
    language: str = "tr",
) -> str | None:
    """Try generating a reply via the configured LLM backend."""

    if not llm_available(settings):
        return None

    messages = _build_messages(user_message, context_text, language)
    provider = settings.get("provider", "off")
    try:
        if provider == "ollama":
            return _call_ollama(messages, settings)
        if provider == "openai":
            return _call_openai(messages, settings)
        if provider == "anthropic":
            return _call_anthropic(messages, settings)
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError):
        return None
    return None


def build_llm_context(
    selected_symptoms: list[str],
    final_prediction: str | None,
    prediction_bundle: Any | None,
    model_accuracies: dict[str, float] | None,
) -> str:
    """Serialize current analysis state for the LLM prompt."""

    lines: list[str] = []
    if selected_symptoms:
        lines.append("Selected symptoms: " + ", ".join(selected_symptoms))
    if final_prediction:
        lines.append(f"Combined prediction: {final_prediction}")
    if prediction_bundle is not None:
        lines.extend(
            [
                f"Decision Tree: {prediction_bundle.decision_tree_prediction}",
                f"Naive Bayes: {prediction_bundle.naive_bayes_prediction}",
                f"Random Forest: {prediction_bundle.random_forest_prediction}",
                f"Logistic Regression: {prediction_bundle.logistic_regression_prediction}",
                f"SVM: {prediction_bundle.svm_prediction}",
                f"XGBoost: {prediction_bundle.xgboost_prediction}",
                f"LightGBM: {prediction_bundle.lightgbm_prediction}",
            ]
        )
        if not prediction_bundle.naive_bayes_probabilities.empty:
            top_probs = prediction_bundle.naive_bayes_probabilities.head(3)
            lines.append(
                "Top Naive Bayes probabilities: "
                + ", ".join(f"{row['prognosis']} ({row['probability_pct']}%)" for _, row in top_probs.iterrows())
            )
    if model_accuracies:
        lines.append(
            "Model accuracies: "
            + ", ".join(f"{name}={round(score * 100, 1)}%" for name, score in model_accuracies.items())
        )
    return "\n".join(lines) if lines else "No analysis context yet."


def _clean_symptom_code(raw: str, valid_symptoms: set[str]) -> str | None:
    """Normalize a model-produced symptom token and validate it."""

    token = raw.strip().strip('"').strip("'")
    token = token.replace(" ", "_").replace("-", "_").lower()
    if token in valid_symptoms:
        return token
    stripped = token.strip("_")
    if stripped in valid_symptoms:
        return stripped
    return None


def extract_symptoms_with_llm(
    user_message: str,
    valid_symptoms: Iterable[str],
    settings: dict[str, Any],
    language: str = "tr",
    max_tokens: int = 256,
) -> tuple[list[str], str]:
    """Use the configured LLM to extract normalized symptom codes from free text.

    Returns (symptom_codes, raw_text). Every returned code is validated against
    ``valid_symptoms``; unknown tokens are dropped. When the backend is not
    configured or the call fails, an empty list is returned so the caller can
    fall back to the rule-based detector.
    """

    if not llm_available(settings):
        return [], ""

    valid_set = set(valid_symptoms)
    if not valid_set:
        return [], ""

    symptom_list = ", ".join(sorted(valid_set))
    instruction = (
        "Extract every symptom mentioned in the user message and return a JSON array "
        f"of EXACT symptom codes chosen ONLY from this list:\n{symptom_list}\n\n"
        "Rules:\n"
        "- Join multi-word codes with underscores exactly as listed (e.g. high_fever).\n"
        "- Do not invent codes; skip anything not in the list.\n"
        "- Return only the JSON array, e.g. [\"high_fever\", \"cough\"].\n"
    )

    messages = [
        {"role": "system", "content": "You are a strict medical symptom extractor. Output JSON only."},
        {"role": "user", "content": f"{instruction}\n\nUser message: {user_message}"},
    ]

    provider = settings.get("provider", "off")
    try:
        if provider == "ollama":
            raw = _call_ollama(messages, settings)
        elif provider == "openai":
            raw = _call_openai(messages, settings)
        elif provider == "anthropic":
            raw = _call_anthropic(messages, settings)
        else:
            return [], ""
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError):
        return [], ""

    parsed = _parse_symptom_json(raw)
    codes: list[str] = []
    for item in parsed:
        code = _clean_symptom_code(item, valid_set)
        if code and code not in codes:
            codes.append(code)
    return codes, (raw or "").strip()


def _parse_symptom_json(raw: str) -> list[str]:
    """Best-effort parse of a model answer that may wrap JSON in prose."""

    if not raw:
        return []
    candidates = []
    try:
        structure_start = raw.find("[")
        structure_end = raw.rfind("]")
        if structure_start != -1 and structure_end > structure_start:
            candidates.append(json.loads(raw[structure_start : structure_end + 1]))
    except (json.JSONDecodeError, ValueError):
        pass
    for candidate in candidates:
        if isinstance(candidate, list):
            items = [
                item
                for item in candidate
                if isinstance(item, str) and item.strip()
            ]
            if items:
                return items
    for line in raw.strip().splitlines():
        stripped = line.strip().strip(",\"]")
        if stripped:
            parts = [part.strip().strip('"') for part in stripped.split(",")]
            return [part for part in parts if part]
    return []
