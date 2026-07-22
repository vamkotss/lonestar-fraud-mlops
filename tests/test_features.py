"""Tests for the Milestone 3 point-in-time feature pipeline.

The headline guarantee is ``test_features_are_future_proof``: no feature value
for a row at time t depends on anything after t. The rest confirm the honest
twins of the M2 leaks behave correctly and that label censoring works.
"""

from __future__ import annotations

import numpy as np

from lonestar import _labeling, audit, features


# --------------------------------------------------------------------------- #
# Spec / hygiene
# --------------------------------------------------------------------------- #
def test_feature_matrix_matches_spec(tx, labels):
    f = features.compute_features(tx, labels)
    assert tuple(f.columns) == features.FEATURE_COLUMNS  # exact set and order
    assert len(f) == len(tx)
    assert not f.isna().any().any()  # a NaN would break the downstream model


# --------------------------------------------------------------------------- #
# THE guarantee: no feature peeks into the future
# --------------------------------------------------------------------------- #
def test_features_are_future_proof(tx, labels):
    """Deleting the latest 20% of rows must not change any surviving feature.

    If any feature used forward-looking data, truncating the future would move
    the values of rows that remain. It must not.
    """
    assert features.is_future_proof(tx, labels, quantile=0.80) is True


# --------------------------------------------------------------------------- #
# Honest twin of Leak C: backward window == audit's backward recompute
# --------------------------------------------------------------------------- #
def test_card_prev_24h_is_the_backward_count(tx, labels):
    f = features.compute_features(tx, labels)
    _, backward = audit._forward_and_backward_counts(
        tx["card_id"].to_numpy(), tx["event_ts"].to_numpy(), horizon_h=24
    )
    assert np.array_equal(f["card_txn_count_prev_24h"].to_numpy(), backward)


def test_velocity_windows_are_monotone(tx, labels):
    """A longer window can never contain fewer events than a shorter one."""
    f = features.compute_features(tx, labels)
    assert (f["card_txn_count_prev_24h"] >= f["card_txn_count_prev_1h"]).all()
    assert (f["card_txn_count_prev_7d"] >= f["card_txn_count_prev_24h"]).all()


# --------------------------------------------------------------------------- #
# Honest twin of Leak B: as-of merchant risk VARIES within a merchant
# --------------------------------------------------------------------------- #
def test_merchant_asof_risk_varies_within_merchant(tx, labels):
    """The M2 leak was constant per merchant; the honest as-of version is not."""
    f = features.compute_features(tx, labels)
    varies = f.assign(m=tx["merchant_id"].to_numpy()).groupby("m")[
        "merchant_fraud_rate_asof"
    ].nunique()
    # Every merchant's risk changes over time as history/chargebacks accrue.
    assert (varies > 1).mean() > 0.9
    # And it is bounded in (0, 1) thanks to the Beta prior (never 0/0).
    assert f["merchant_fraud_rate_asof"].between(0, 1).all()


def test_merchant_risk_starts_near_the_prior(tx, labels):
    """The very first transaction on a merchant has no known chargebacks yet,
    so its as-of risk should equal the Beta prior mean."""
    f = features.compute_features(tx, labels)
    prior_mean = features._PRIOR_ALPHA / (features._PRIOR_ALPHA + features._PRIOR_BETA)
    first_rows = f["merchant_txns_seen"] == 0
    assert first_rows.any()
    assert np.allclose(f.loc[first_rows, "merchant_fraud_rate_asof"], prior_mean)


# --------------------------------------------------------------------------- #
# Label censoring by reported_at
# --------------------------------------------------------------------------- #
def test_label_censoring_hides_unreported_fraud(tx, labels):
    """At an early as_of, fewer frauds are known than the eventual total."""
    early = tx["event_ts"].min() + np.timedelta64(120, "D")
    frame_early = features.build_training_frame(tx, labels, as_of=early)
    eventual = _labeling.attach_row_label(tx, labels).mean()
    assert frame_early["is_fraud"].mean() < eventual
    # Every row in the censored frame occurred on or before the cutoff.
    assert (frame_early["event_ts"] <= early).all()


# --------------------------------------------------------------------------- #
# The honest signal exists (and the diagnostic inversion is real)
# --------------------------------------------------------------------------- #
def test_robust_features_carry_honest_signal(tx, labels):
    auc = features.temporal_split_auc(tx, labels, cols=features.ROBUST_FEATURE_COLUMNS)
    # Believable and leak-free -- nowhere near the M2 leaky ~1.0.
    assert 0.65 < auc < 0.90


def test_merchant_asof_inversion_is_the_label_delay_finding(tx, labels):
    """Documented finding: under label delay, as-of merchant risk is inverted on
    the drift window (new attackers look clean). This test pins that phenomenon."""
    auc = features.temporal_split_auc(tx, labels, cols=features.DIAGNOSTIC_FEATURE_COLUMNS)
    assert auc < 0.5
