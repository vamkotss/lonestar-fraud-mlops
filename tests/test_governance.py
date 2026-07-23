"""Tests for the Milestone 10 governance layer.

Two things are worth testing about documentation:
  * that the completeness gate actually catches a missing artifact, and
  * that the generated model card reports the numbers really in the artifacts
    (and degrades gracefully when a stage has not been run yet).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lonestar import governance

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# The artifact completeness gate
# --------------------------------------------------------------------------- #
def test_repo_has_every_required_artifact():
    """The governance gate CI runs: every required doc/artifact must exist."""
    result = governance.check_artifacts(_REPO_ROOT)
    assert result.ok, f"missing governance artifacts: {result.missing}"


def test_check_detects_a_missing_artifact(tmp_path):
    """An empty tree must fail the gate, listing what is absent and why."""
    result = governance.check_artifacts(tmp_path)
    assert not result.ok
    assert len(result.missing) == len(governance.REQUIRED_ARTIFACTS)
    for _rel, why in result.missing:
        assert why, "every required artifact needs a stated reason"


def test_every_required_artifact_has_a_rationale():
    """A required file with no stated purpose is cargo cult; forbid it."""
    for rel, why in governance.REQUIRED_ARTIFACTS:
        assert rel and why
        assert len(why) > 10


# --------------------------------------------------------------------------- #
# Model card generation
# --------------------------------------------------------------------------- #
def test_gather_reads_repo_artifacts():
    data = governance.gather(_REPO_ROOT)
    assert data["card"] is not None, "models/model_card.json should be readable"
    assert data["policy"] is not None, "models/decision_policy.json should be readable"


def test_model_card_has_the_required_sections():
    card = governance.render_model_card(governance.gather(_REPO_ROOT))
    for heading in (
        "# Model Card",
        "## Model details",
        "## Intended use",
        "## Performance",
        "## Data leakage audit",
        "## Decision policy",
        "## Monitoring",
        "## Retraining",
        "## Limitations and known risks",
        "## Fairness and ethical considerations",
        "## Reproducing these numbers",
    ):
        assert heading in card, f"model card is missing section: {heading}"


def test_model_card_reports_the_real_numbers(tmp_path):
    """Numbers in the card must come from the artifacts, not be hardcoded."""
    (tmp_path / "models").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "models" / "model_card.json").write_text(
        json.dumps(
            {
                "model_kind": "xgb",
                "feature_columns": ["a", "b", "c"],
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "holdout_metrics": {"pr_auc": 0.4242, "roc_auc": 0.8181},
                "training_window": {"cutoff": "2025-01-01", "n_train": 12345},
                "test_prevalence": 0.0053,
                "primary_metric": "pr_auc",
            }
        )
    )
    (tmp_path / "reports" / "leakage_audit.json").write_text(
        json.dumps({"auc_with_leaks": 1.0, "auc_without_leaks": 0.79, "auc_gap": 0.21})
    )

    card = governance.render_model_card(governance.gather(tmp_path))
    assert "0.4242" in card  # the PR-AUC we wrote
    assert "0.8181" in card  # the ROC-AUC we wrote
    assert "+0.2100" in card  # the leakage gap, formatted
    assert "3 point-in-time features" in card


def test_model_card_degrades_gracefully_on_a_fresh_clone(tmp_path):
    """With no artifacts at all the card still renders, marking gaps honestly."""
    card = governance.render_model_card(governance.gather(tmp_path))
    assert "# Model Card" in card
    assert "_not yet generated_" in card


def test_model_card_documents_limitations():
    """A model card without limitations is marketing, not governance."""
    card = governance.render_model_card(governance.gather(_REPO_ROOT))
    # The two honest limitations this project is built around.
    assert "label delay" in card.lower()
    assert "synthetic" in card.lower()


# --------------------------------------------------------------------------- #
# The runbook is a required, non-trivial artifact
# --------------------------------------------------------------------------- #
def test_runbook_covers_the_operational_gap():
    runbook = (_REPO_ROOT / "docs" / "runbook.md").read_text(encoding="utf-8")
    # The scenario that cannot be solved by modelling must be addressed.
    assert "Rollback" in runbook or "rollback" in runbook
    assert "30" in runbook and "60" in runbook  # the label delay window
    for section in ("Drift alert fired", "Retrain", "Escalation"):
        assert section.lower() in runbook.lower()


@pytest.mark.parametrize("doc", ["model_card.json", "decision_policy.json"])
def test_machine_readable_artifacts_are_valid_json(doc):
    payload = json.loads((_REPO_ROOT / "models" / doc).read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and payload
