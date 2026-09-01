"""Model artifact versioning: schema version, checksums and a manifest.

Binary artifacts under ``models/`` are reproducible from the training data via
``retrain_all_models``, so they are intentionally not source-controlled. This
module gives every artifact a schema version plus a SHA-256 checksum and keeps
a single ``models/manifest.json`` describing the current artifact set (data
fingerprint, training timestamp, size and hash). The manifest is rewritten
after every full retrain and can be used to verify deployments or detect
corrupted/outdated bundles.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Bump whenever the persisted bundle layout changes (joblib dict keys, model
#: hyperparameters, evaluation semantics) so stale artifacts are rejected.
MODEL_SCHEMA_VERSION = 2

MANIFEST_NAME = "manifest.json"


def models_directory() -> Path:
    """Return the absolute path of the models directory."""

    return Path(__file__).resolve().parent.parent / "models"


def compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a binary artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_info(path: Path) -> dict[str, Any]:
    """Describe a single persisted artifact for the manifest."""

    info: dict[str, Any] = {
        "artifact": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": compute_file_sha256(path),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    try:
        bundle_metadata = _bundle_metadata(path)
        if bundle_metadata:
            info["training_file"] = bundle_metadata.get("training_file")
            info["data_fingerprint"] = bundle_metadata.get("data_fingerprint")
            info["trained_at"] = bundle_metadata.get("trained_at")
            info["data_rows"] = bundle_metadata.get("data_rows")
            info["class_count"] = bundle_metadata.get("class_count")
    except Exception:
        pass
    return info


def _bundle_metadata(path: Path) -> dict[str, Any] | None:
    """Read the metadata dict embedded in a joblib bundle, if present."""

    import joblib

    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    if isinstance(bundle, dict) and isinstance(bundle.get("metadata"), dict):
        return bundle["metadata"]
    return None


def write_models_manifest(training_file: str = "Training.csv") -> Path:
    """Scan ``models/`` and persist ``manifest.json`` with artifact metadata.

    A hash of the current training data is included so a deployment can verify
    that its artifact set matches the dataset it was trained on.
    """

    from src.model_metadata import compute_fingerprint

    try:
        fingerprint = compute_fingerprint(training_file)
    except Exception:
        fingerprint = {}

    directory = models_directory()
    artifacts = [
        _artifact_info(path)
        for path in sorted(directory.glob("*.pkl"))
        if path.name != MANIFEST_NAME
    ]
    manifest = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_file": training_file,
        "data_fingerprint": fingerprint.get("data_fingerprint"),
        "data_rows": fingerprint.get("data_rows"),
        "class_count": fingerprint.get("class_count"),
        "artifacts": artifacts,
    }
    manifest_path = directory / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def read_models_manifest() -> dict[str, Any] | None:
    """Load ``models/manifest.json`` or return None when it does not exist."""

    manifest_path = models_directory() / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def manifest_status(training_file: str = "Training.csv") -> dict[str, Any]:
    """Compare the manifest against the current data and artifact state."""

    manifest = read_models_manifest()
    directory = models_directory()
    artifact_names = sorted(path.name for path in directory.glob("*.pkl") if path.name != MANIFEST_NAME)
    if manifest is None:
        return {"present": False, "artifacts": artifact_names}
    from src.model_metadata import compute_fingerprint

    try:
        current = compute_fingerprint(training_file)
    except Exception:
        current = {}
    manifest_artifacts = {entry["artifact"] for entry in manifest.get("artifacts", [])}
    return {
        "present": True,
        "schema_version": manifest.get("model_schema_version"),
        "schema_current": manifest.get("model_schema_version") == MODEL_SCHEMA_VERSION,
        "data_fingerprint": manifest.get("data_fingerprint"),
        "data_fresh": manifest.get("data_fingerprint") == current.get("data_fingerprint"),
        "training_file": manifest.get("training_file"),
        "artifacts": artifact_names,
        "artifacts_complete": set(artifact_names) == manifest_artifacts,
    }