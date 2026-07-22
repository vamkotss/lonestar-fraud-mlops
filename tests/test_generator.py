"""Contract tests for the transaction/label generator.

Each test pins one property the downstream MLOps work depends on. If any of
these drifts, later milestones (leakage audit, temporal splits, drift monitor)
would silently break -- so they are guarded here, at the source of the data.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lonestar import _leaks
from lonestar.generation import GenConfig, generate

RING_ONSET = 14
BASE_RATE = 0.0015


# --------------------------------------------------------------------------- #
# CI mode
# --------------------------------------------------------------------------- #
def test_ci_mode_is_smaller_than_full(monkeypatch):
    """LS_CI_MODE must select a small dataset; unset must select the full one."""
    monkeypatch.setenv("LS_CI_MODE", "1")
    ci = GenConfig.from_env()
    monkeypatch.delenv("LS_CI_MODE", raising=False)
    full = GenConfig.from_env()
    assert ci.ci_mode is True
    assert full.ci_mode is False
    assert ci.n_purchases < full.n_purchases


def test_full_mode_targets_about_5m_rows():
    """The full config is sized so the transactions table lands near ~5M rows."""
    full = GenConfig(**{**asdict(GenConfig.from_env()), "ci_mode": False, "n_purchases": 2_700_000})
    approx_rows = full.n_purchases * (1 + full.capture_rate)
    assert 4_500_000 <= approx_rows <= 5_500_000


# --------------------------------------------------------------------------- #
# Schema / ADR: labels live in a separate table
# --------------------------------------------------------------------------- #
def test_labels_are_separated_from_transactions(tx, labels):
    # No fraud/label column may leak onto the transactions table.
    for banned in ("is_fraud", "fraud", "label", "chargeback", "reported_at"):
        assert banned not in tx.columns
    # The label table carries the late-arriving signal explicitly.
    for col in ("auth_code", "is_fraud", "reported_at", "event_ts", "fraud_type"):
        assert col in labels.columns
    assert labels["is_fraud"].all()  # a chargeback table is fraud-only by definition


# --------------------------------------------------------------------------- #
# Imbalance
# --------------------------------------------------------------------------- #
def test_class_imbalance_rate(tx, labels):
    n_purchases = int((tx["event_type"] == "AUTH").sum())
    overall = len(labels) / n_purchases
    # ~0.15% base + a ring uplift after month 14 -> total sits in this band.
    assert 0.0015 <= overall <= 0.0035, f"overall fraud rate {overall:.5f} out of band"


def test_pre_onset_rate_matches_baseline(labels, tx):
    auth = tx[tx["event_type"] == "AUTH"].copy()
    auth["month"] = (auth["event_ts"].dt.year - 2024) * 12 + auth["event_ts"].dt.month
    fraud_auth = set(labels["auth_code"])
    auth["is_fraud"] = auth["auth_code"].isin(fraud_auth)
    pre = auth.loc[auth["month"] < RING_ONSET, "is_fraud"].mean()
    assert 0.0010 <= pre <= 0.0022, f"pre-onset rate {pre:.5f} should hug {BASE_RATE}"


# --------------------------------------------------------------------------- #
# Label delay
# --------------------------------------------------------------------------- #
def test_label_delay_distribution(labels):
    delay_days = (labels["reported_at"] - labels["event_ts"]).dt.days
    assert delay_days.min() >= 30
    assert delay_days.max() <= 60
    assert 42 <= delay_days.mean() <= 48  # triangular(30,45,60) -> mean ~45


# --------------------------------------------------------------------------- #
# Drift onset (the month-14 fraud ring)
# --------------------------------------------------------------------------- #
def test_ring_confined_to_and_present_after_onset(labels):
    lab = labels.copy()
    lab["month"] = (lab["event_ts"].dt.year - 2024) * 12 + lab["event_ts"].dt.month
    ring = lab[lab["fraud_type"] == "RING"]
    assert len(ring) > 0, "expected a fraud ring"
    assert ring["month"].min() >= RING_ONSET, "ring leaked before onset month"


def test_fraud_rate_jumps_at_onset(tx, labels):
    auth = tx[tx["event_type"] == "AUTH"].copy()
    auth["month"] = (auth["event_ts"].dt.year - 2024) * 12 + auth["event_ts"].dt.month
    fraud_auth = set(labels["auth_code"])
    auth["is_fraud"] = auth["auth_code"].isin(fraud_auth)
    pre = auth.loc[auth["month"] < RING_ONSET, "is_fraud"].mean()
    post = auth.loc[auth["month"] >= RING_ONSET, "is_fraud"].mean()
    assert post > 1.5 * pre, f"expected drift: post {post:.5f} vs pre {pre:.5f}"


def test_ring_shifts_feature_distribution(tx, labels):
    """The ring should visibly move features, not just the rate (covariate drift)."""
    fraud_auth = set(labels[labels["fraud_type"] == "RING"]["auth_code"])
    ring_rows = tx[tx["auth_code"].isin(fraud_auth)]
    # Ring is e-commerce and disproportionately foreign -> both far above baseline.
    assert (ring_rows["entry_mode"] == "ECOM").mean() > 0.9
    assert (ring_rows["pos_country"] != "US").mean() > 0.5


# --------------------------------------------------------------------------- #
# Leak A -- post-outcome field
# --------------------------------------------------------------------------- #
def test_leak_a_dispute_code_perfectly_separates(tx, row_fraud):
    has_code = (tx["dispute_reason_code"] != "NONE").to_numpy()
    agreement = (has_code == row_fraud).mean()
    assert agreement == 1.0, "dispute_reason_code must be a perfect (leaky) proxy for fraud"


# --------------------------------------------------------------------------- #
# Leak B -- look-ahead target encoding
# --------------------------------------------------------------------------- #
def test_leak_b_merchant_lifetime_rate_leaks(tx, row_fraud):
    auc = roc_auc_score(row_fraud, tx["merchant_fraud_rate_lifetime"].to_numpy())
    assert auc > 0.70, f"lifetime rate should be strongly predictive (leaky); AUC={auc:.3f}"
    # Proof of look-ahead: an honest as-of feature would vary over time within a
    # merchant. A single value per merchant can only be a global (future-inclusive)
    # aggregate.
    per_merchant = tx.groupby("merchant_id")["merchant_fraud_rate_lifetime"].nunique()
    assert (per_merchant > 1).sum() == 0, "value must be constant within merchant_id (global)"
    assert (tx.groupby("merchant_id")["merchant_fraud_rate_lifetime"].first() > 0).sum() > 0


# --------------------------------------------------------------------------- #
# Leak C -- temporal look-ahead window
# --------------------------------------------------------------------------- #
def test_leak_c_next_24h_is_forward_looking(tx):
    def forward_count(df, i):
        r = df.iloc[i]
        same = df[df["card_id"] == r["card_id"]]
        window = (same["event_ts"] > r["event_ts"]) & (
            same["event_ts"] <= r["event_ts"] + np.timedelta64(24, "h")
        )
        return int(window.sum())

    rng = np.random.default_rng(7)
    sample = rng.choice(len(tx), size=40, replace=False)
    for i in sample:
        assert int(tx.iloc[i]["card_txn_count_next_24h"]) == forward_count(tx, int(i))
    # And the feature is actually used (not trivially all-zero).
    assert (tx["card_txn_count_next_24h"] > 0).sum() > 0


def test_leak_registry_columns_present(tx):
    for col in _leaks.LEAK_COLUMNS:
        assert col in tx.columns


# --------------------------------------------------------------------------- #
# Realistic mess
# --------------------------------------------------------------------------- #
def test_missing_mcc_rate(tx):
    miss = tx["mcc"].isna().mean()
    assert 0.03 <= miss <= 0.05, f"missing MCC rate {miss:.3f} off target ~4%"


def test_chaotic_merchant_names(tx):
    names_per_id = tx.groupby("merchant_id")["merchant_name"].nunique()
    assert names_per_id.max() > 1, "one merchant should render under many descriptors"
    # A naive GROUP BY name fragments merchants badly vs the clean id count.
    assert tx["merchant_name"].nunique() > 5 * tx["merchant_id"].nunique()


def test_duplicate_auth_capture_pairs(tx):
    types = set(tx["event_type"].unique())
    assert {"AUTH", "CAPTURE"} <= types
    both = tx.groupby("auth_code")["event_type"].nunique()
    assert (both == 2).sum() > 0, "expected auth/capture duplicate pairs"
    # De-duplicating to the purchase grain shrinks the row count.
    assert tx["auth_code"].nunique() < len(tx)


def test_timestamps_are_timezone_naive(tx, labels):
    assert tx["event_ts"].dt.tz is None
    assert labels["reported_at"].dt.tz is None
    assert labels["event_ts"].dt.tz is None


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_is_deterministic():
    cfg = GenConfig(**{**asdict(GenConfig.from_env()), "n_purchases": 20_000, "ci_mode": True})
    a_tx, a_lab, _ = generate(cfg)
    b_tx, b_lab, _ = generate(cfg)
    assert len(a_tx) == len(b_tx)
    assert len(a_lab) == len(b_lab)
    assert a_tx["transaction_id"].tolist()[:50] == b_tx["transaction_id"].tolist()[:50]
