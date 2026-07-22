"""Tests for the Milestone 5 decision layer.

These use a small synthetic scored frame (label + probability + amount + segment)
so the cost/threshold logic is checked fast and deterministically. The real CLI
runs end-to-end on the actual model artifacts in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lonestar import decisions


@pytest.fixture
def scored() -> pd.DataFrame:
    """A realistic-ish scored frame: rare fraud, ECOM riskier, scores track truth."""
    rng = np.random.default_rng(0)
    n = 6000
    amount = rng.lognormal(3.5, 1.0, n)
    entry = rng.choice(["CHIP", "ECOM", "SWIPE"], n, p=[0.6, 0.3, 0.1])
    base = np.where(entry == "ECOM", 0.03, 0.005)  # ECOM ~6x riskier
    is_fraud = (rng.random(n) < base).astype(int)
    # Score correlates with the label (a useful model) but is noisy.
    proba = np.clip(0.55 * is_fraud + rng.normal(0.15, 0.15, n), 0.0, 1.0)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"is_fraud": is_fraud, "proba": proba, "amount": amount, "entry_mode": entry, "event_ts": ts}
    )


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
def test_cost_model_asymmetry():
    cost = decisions.CostModel(fp_fixed=5.0, fp_rate=0.02)
    amt = np.array([100.0, 500.0])
    # A missed fraud costs the whole amount.
    assert list(cost.false_negative_cost(amt)) == [100.0, 500.0]
    # A false decline costs only fixed + a small fraction.
    assert list(cost.false_positive_cost(amt)) == [7.0, 15.0]
    # And the miss is always the more expensive mistake here.
    assert (cost.false_negative_cost(amt) > cost.false_positive_cost(amt)).all()


def test_expected_cost_hand_example():
    cost = decisions.CostModel(fp_fixed=5.0, fp_rate=0.0)
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.1, 0.9, 0.8, 0.2])
    amount = np.array([100.0, 100.0, 100.0, 100.0])
    # Threshold 0.5: declines rows 1 and 2. Row0 approved fraud -> FN cost 100.
    # Row1 declined legit -> FP cost 5. Row2 declined fraud -> ok. Row3 approved legit -> ok.
    assert decisions.expected_cost(y, proba, amount, 0.5, cost) == 105.0


# --------------------------------------------------------------------------- #
# Optimal threshold beats the naive default on the data it is fit on
# --------------------------------------------------------------------------- #
def test_optimal_threshold_beats_default(scored):
    cost = decisions.CostModel()
    y = scored["is_fraud"].to_numpy()
    proba = scored["proba"].to_numpy()
    amount = scored["amount"].to_numpy()
    t_star, cost_star = decisions.optimal_threshold(y, proba, amount, cost)
    cost_05 = decisions.expected_cost(y, proba, amount, 0.5, cost)
    # By construction the optimum is no worse than the default 0.5.
    assert cost_star <= cost_05
    assert 0.0 <= t_star <= 1.0


# --------------------------------------------------------------------------- #
# Per-segment policy is never worse than a single global threshold (on fit data)
# --------------------------------------------------------------------------- #
def test_per_segment_beats_global_on_fit_data(scored):
    cost = decisions.CostModel()
    policy = decisions.fit_segment_policy(scored, cost)

    assert set(policy["segment_thresholds"]) == set(scored["entry_mode"].unique())

    y = scored["is_fraud"].to_numpy()
    amount = scored["amount"].to_numpy()
    proba = scored["proba"].to_numpy()

    global_cost = decisions.expected_cost(y, proba, amount, policy["global_threshold"], cost)
    seg_decline = decisions.apply_policy(scored, policy)
    fn = (~seg_decline) & (y == 1)
    fp = seg_decline & (y == 0)
    seg_cost = decisions.expected_cost_from_masks(fn, fp, amount, cost)

    # Segmenting can only help (or tie) on the data the thresholds were fit on.
    assert seg_cost <= global_cost + 1e-6


def test_apply_policy_respects_thresholds(scored):
    cost = decisions.CostModel()
    policy = decisions.fit_segment_policy(scored, cost)
    decline = decisions.apply_policy(scored, policy)
    # Recompute expected decisions from the per-segment thresholds directly.
    seg = scored["entry_mode"].astype(str).to_numpy()
    expected = scored["proba"].to_numpy() >= np.array(
        [policy["segment_thresholds"][s] for s in seg]
    )
    assert np.array_equal(decline, expected)


# --------------------------------------------------------------------------- #
# Economics evaluation
# --------------------------------------------------------------------------- #
def test_evaluate_quantifies_savings(scored):
    cost = decisions.CostModel()
    # Fit on the first 70% of time, evaluate on the rest (temporal discipline).
    cutoff = scored["event_ts"].quantile(0.70)
    train = scored[scored["event_ts"] < cutoff]
    test = scored[scored["event_ts"] >= cutoff]
    policy = decisions.fit_segment_policy(train, cost)
    econ = decisions.evaluate(test, policy, cost)

    # Approve-all catches no fraud and its cost equals all fraud dollars.
    assert econ["approve_all"]["fraud_dollars_caught"] == 0.0
    assert econ["approve_all"]["savings_vs_approve_all"] == 0.0
    # Every active policy saves real money against doing nothing.
    for key in ("threshold_0.5", "global_optimal", "per_segment"):
        assert econ[key]["savings_vs_approve_all"] > 0
    # And catches most of the fraud dollars.
    assert econ["per_segment"]["fraud_recall_dollars"] > 0.5
