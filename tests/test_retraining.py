"""Tests for the Milestone 9 challenger/champion retraining cycle.

The headline tests:
  * ``test_training_never_uses_immature_labels`` -- the constraint the whole
    design exists to respect.
  * ``test_stale_champion_is_replaced`` / ``test_no_new_signal_holds`` -- the
    pipeline promotes when there is real new signal and refuses when there is not.
  * the guardrail tests -- a challenger cannot win on a handful of positives, on a
    marginal delta, or by sacrificing a channel.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lonestar import features, retraining

_START = pd.Timestamp("2024-01-01")


# --------------------------------------------------------------------------- #
# Windows and label maturity
# --------------------------------------------------------------------------- #
def test_maturity_cutoff_excludes_recent_transactions():
    as_of = pd.Timestamp("2025-06-01")
    cutoff = retraining.maturity_cutoff(as_of)
    assert (as_of - cutoff).days == retraining.LABEL_MATURITY_DAYS


def test_training_windows_are_ordered_and_disjoint():
    as_of = pd.Timestamp("2025-06-01")
    w = retraining.training_windows(as_of)
    # train < eval_start == eval_end - EVAL_WINDOW < as_of
    assert w["train_end"] == w["eval_start"]
    assert w["eval_start"] < w["eval_end"]
    assert w["eval_end"] < as_of
    assert (w["eval_end"] - w["eval_start"]).days == retraining.EVAL_WINDOW_DAYS
    # Nothing in the eval window is younger than the maturity horizon.
    assert (as_of - w["eval_end"]).days >= retraining.LABEL_MATURITY_DAYS


def test_training_never_uses_immature_labels(tx, labels):
    """THE constraint: no training row is newer than the maturity horizon.

    Training on rows whose chargebacks have not arrived would silently label live
    fraud as legitimate and teach the model precisely the wrong thing.
    """
    as_of = tx["event_ts"].max()
    w = retraining.training_windows(as_of)
    frame = features.build_training_frame(tx, labels, as_of=as_of)
    _, info = retraining.train_as_of(frame, w["train_end"])
    train_rows = frame[frame["event_ts"] < w["train_end"]]
    assert len(train_rows) == info["n_rows"]
    # Every training row is old enough for its label to have matured.
    assert (as_of - train_rows["event_ts"].max()).days >= retraining.LABEL_MATURITY_DAYS


def test_label_censoring_hides_unreported_chargebacks(tx, labels):
    """At an earlier as_of, fewer chargebacks are visible to the trainer."""
    early = tx["event_ts"].min() + np.timedelta64(200, "D")
    late = tx["event_ts"].max()
    n_early = features.build_training_frame(tx, labels, as_of=early)["is_fraud"].sum()
    n_late = features.build_training_frame(tx, labels, as_of=late)["is_fraud"].sum()
    assert n_early < n_late


def test_train_as_of_needs_positives(tx, labels):
    frame = features.build_training_frame(tx, labels)
    with pytest.raises(ValueError):
        # A window before any fraud has been reported has nothing to learn from.
        retraining.train_as_of(frame, frame["event_ts"].min())


# --------------------------------------------------------------------------- #
# Promotion guardrails
# --------------------------------------------------------------------------- #
def _eval(pr_auc: float, positives: int = 100, segments: dict | None = None) -> dict:
    return {
        "overall": {"pr_auc": pr_auc, "roc_auc": 0.8},
        "per_segment": segments or {},
        "n_rows": 10000,
        "n_positives": positives,
    }


def test_promotes_on_material_improvement():
    d = retraining.promotion_decision(_eval(0.40), _eval(0.60))
    assert d.promote
    assert d.relative_improvement == pytest.approx(0.5, abs=0.01)


def test_holds_on_marginal_improvement():
    """A 1% gain is noise, not evidence -- the champion stays."""
    d = retraining.promotion_decision(_eval(0.500), _eval(0.505))
    assert not d.promote


def test_holds_when_challenger_is_worse():
    d = retraining.promotion_decision(_eval(0.60), _eval(0.40))
    assert not d.promote


def test_holds_on_too_few_positives():
    """A PR-AUC computed on a handful of fraud cases cannot justify a deployment."""
    few = retraining.MIN_EVAL_POSITIVES - 1
    d = retraining.promotion_decision(_eval(0.40, positives=few), _eval(0.90, positives=few))
    assert not d.promote
    assert any("fraud cases" in r for r in d.reasons)


def test_holds_when_a_segment_regresses():
    """An average can improve while a whole channel is quietly sacrificed."""
    champ = _eval(0.50, segments={"ECOM": {"pr_auc": 0.60}, "CHIP": {"pr_auc": 0.40}})
    chall = _eval(0.70, segments={"ECOM": {"pr_auc": 0.40}, "CHIP": {"pr_auc": 0.90}})
    d = retraining.promotion_decision(champ, chall)
    assert not d.promote
    assert "ECOM" in d.segment_regressions


# --------------------------------------------------------------------------- #
# End-to-end cycles
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def stale_champion(tx, labels):
    """A champion trained before the fraud ring appeared."""
    as_of = _START + pd.DateOffset(months=12)
    frame = features.build_training_frame(tx, labels, as_of=as_of)
    model, _ = retraining.train_as_of(frame, retraining.training_windows(as_of)["train_end"])
    return model


def test_stale_champion_is_replaced(tx, labels, stale_champion):
    """A pre-ring champion has no idea what the attack looks like; retraining on
    matured post-ring labels should decisively beat it."""
    rep = retraining.run_cycle(tx, labels, stale_champion)
    rep.pop("challenger")
    assert rep["decision"]["promote"]
    assert rep["challenger_eval"]["overall"]["pr_auc"] > rep["champion_eval"]["overall"]["pr_auc"]


def test_no_new_signal_holds(tx, labels):
    """Retraining against a champion trained on the SAME data should not promote:
    there is no new information, so the guardrails must refuse."""
    as_of = tx["event_ts"].max()
    frame = features.build_training_frame(tx, labels, as_of=as_of)
    champion, _ = retraining.train_as_of(frame, retraining.training_windows(as_of)["train_end"])
    rep = retraining.run_cycle(tx, labels, champion)
    rep.pop("challenger")
    assert not rep["decision"]["promote"]


def test_cycle_report_shape(tx, labels, stale_champion):
    rep = retraining.run_cycle(tx, labels, stale_champion)
    rep.pop("challenger")
    for key in ("as_of", "windows", "champion_eval", "challenger_eval", "decision"):
        assert key in rep
    assert rep["label_maturity_days"] == retraining.LABEL_MATURITY_DAYS
    md = retraining._render_markdown(rep)
    assert "# Retraining Cycle Report" in md


# --------------------------------------------------------------------------- #
# The Airflow DAG: structural check without requiring Airflow
# --------------------------------------------------------------------------- #
def test_dag_file_parses_and_defines_expected_tasks():
    """Airflow is intentionally not a CI dependency, so validate the DAG by parsing.

    This catches syntax errors and missing task callables -- the failures that
    actually break a deployment -- without installing a heavy scheduler.
    """
    dag_path = Path(__file__).resolve().parents[1] / "dags" / "lonestar_retraining_dag.py"
    assert dag_path.exists(), "retraining DAG file is missing"
    tree = ast.parse(dag_path.read_text(encoding="utf-8"))
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for expected in (
        "task_check_drift",
        "task_should_retrain",
        "task_run_retraining_cycle",
        "task_skip_retraining",
    ):
        assert expected in functions, f"DAG is missing task callable {expected}"


def test_dag_module_imports_without_airflow():
    """The module must be importable even when Airflow is absent."""
    import importlib.util

    dag_path = Path(__file__).resolve().parents[1] / "dags" / "lonestar_retraining_dag.py"
    spec = importlib.util.spec_from_file_location("ls_dag", dag_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Branch logic is pure Python and testable directly, Airflow or not.
    assert hasattr(module, "task_should_retrain")
