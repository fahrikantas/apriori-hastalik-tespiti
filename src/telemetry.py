"""Local, privacy-preserving telemetry for prediction quality monitoring.

Every prediction is appended as one JSONL line under ``data/telemetry/``.
Only symptom codes (stable identifiers), model outputs and aggregate
indicators are recorded — no free text, no patient identifiers, and nothing
leaves the machine. The disagreement summary highlights the symptom
combinations on which the models do not agree, which is the most actionable
signal for data-driven improvement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TELEMETRY_DIR_NAME = "data/telemetry"
TELEMETRY_FILE_NAME = "predictions.jsonl"
DISAGREEMENT_THRESHOLD = 3


def telemetry_path() -> Path:
    """Return the JSONL telemetry file path, creating the directory if needed."""

    path = Path(__file__).resolve().parent.parent / TELEMETRY_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path / TELEMETRY_FILE_NAME


def model_agreement(predictions: list[str]) -> dict[str, Any]:
    """Compute agreement stats over the per-model disease predictions."""

    total = len(predictions)
    unique = len(set(predictions))
    top_prediction = max(set(predictions), key=predictions.count) if predictions else None
    top_count = predictions.count(top_prediction) if top_prediction else 0
    return {
        "model_count": total,
        "unique_predictions": unique,
        "agreement_fraction": round(top_count / total, 3) if total else 0.0,
        "disagreement": unique > 1 and top_count < DISAGREEMENT_THRESHOLD,
    }


def log_prediction(
    training_file: str,
    symptoms: list[str],
    model_predictions: dict[str, str],
    final_prediction: str | None = None,
    confidence_level: str | None = None,
    ood: dict[str, Any] | None = None,
    agreement: dict[str, Any] | None = None,
) -> None:
    """Append one prediction record to the local telemetry log."""

    agreement = agreement or model_agreement(list(model_predictions.values()))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_file": training_file,
        "symptoms": sorted(symptoms),
        "symptom_count": len(symptoms),
        "model_predictions": model_predictions,
        "final_prediction": final_prediction,
        "confidence_level": confidence_level,
        "ood": ood,
        "agreement": agreement,
    }
    line = json.dumps(record, ensure_ascii=True, sort_keys=True)
    with telemetry_path().open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_records(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the latest telemetry records (newest first)."""

    path = telemetry_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except Exception:
                continue
    if limit is not None:
        records = records[-limit:]
    return list(reversed(records))


def summarize_disagreements(limit: int = 500) -> dict[str, Any]:
    """Summarize recent predictions: volume, agreement and top disagreement combos."""

    records = read_records(limit)
    total = len(records)
    disagreements = [
        record for record in records if (record.get("agreement") or {}).get("disagreement")
    ]
    combo_counts: dict[str, tuple[int, dict[str, Any]]] = {}
    for record in disagreements:
        key = "|".join(sorted(record.get("symptoms", [])))
        if not key:
            continue
        count, sample = combo_counts.get(key, (0, {}))
        combo_counts[key] = (count + 1, sample or record)
    top_combos = [
        {
            "symptoms": sample.get("symptoms", []),
            "count": count,
            "predictions": sorted(set((sample.get("model_predictions") or {}).values())),
        }
        for key, (count, sample) in sorted(
            combo_counts.items(), key=lambda item: item[1][0], reverse=True
        )[:10]
    ]
    return {
        "record_count": total,
        "disagreement_count": len(disagreements),
        "disagreement_rate": round(len(disagreements) / total, 3) if total else 0.0,
        "top_disagreements": top_combos,
    }


def clear_telemetry() -> int:
    """Delete the telemetry log and return how many records were removed."""

    path = telemetry_path()
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    path.unlink()
    return count