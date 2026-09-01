# -*- coding: utf-8 -*-
"""Model versioning and manifest tests."""

from __future__ import annotations

from src.versioning import (
    MODEL_SCHEMA_VERSION,
    compute_file_sha256,
    manifest_status,
    read_models_manifest,
    write_models_manifest,
)


def test_schema_version_is_defined():
    assert MODEL_SCHEMA_VERSION >= 1


def test_manifest_roundtrip():
    manifest_path = write_models_manifest("Training.csv")
    assert manifest_path.exists()
    manifest = read_models_manifest()
    assert manifest is not None
    assert manifest["model_schema_version"] == MODEL_SCHEMA_VERSION
    names = {entry["artifact"] for entry in manifest["artifacts"]}
    assert "decision_tree.pkl" in names
    assert all(entry["sha256"] for entry in manifest["artifacts"])


def test_manifest_status_reflects_artifacts():
    write_models_manifest("Training.csv")
    status = manifest_status("Training.csv")
    assert status["present"] is True
    assert status["schema_current"] is True
    assert status["artifacts_complete"] is True
    assert "decision_tree.pkl" in status["artifacts"]


def test_sha256_is_stable():
    from src.utils import resolve_model_path

    path = resolve_model_path("decision_tree.pkl")
    assert compute_file_sha256(path) == compute_file_sha256(path)