"""Tests for the Milestone 2 leakage audit.

These prove three things the interview story rests on:
  1. Each detector fires on its own leak (and the univariate screen alone does NOT).
  2. The temporal split is genuinely past-vs-future (no overlap, correct ordering).
  3. Removing the leaks collapses a fantasy AUC (~1.0) to an honest one (~0.8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lonestar import _labeling, audit


# --------------------------------------------------------------------------- #
# Row label helper
# --------------------------------------------------------------------------- #
def test_attach_row_label_matches_auth_code_membership(tx, labels, row_fraud):
    """The audit's label must equal the conftest's independently-computed one."""
    y = _labeling.attach_row_label(tx, labels).to_numpy()
    assert y.dtype == bool
    assert np.array_equal(y, row_fraud)
    # Rare-event problem: fraud rate is a fraction of a percent.
    assert 0.0010 < y.mean() < 0.0060


def test_attach_row_label_respects_as_of(tx, labels):
    """With an early as_of, fewer chargebacks are 'known' (they report late)."""
    early = tx["event_ts"].min() + pd.Timedelta(days=90)
    known_early = _labeling.attach_row_label(tx, labels, as_of=early).sum()
    known_all = _labeling.attach_row_label(tx, labels).sum()
    # Some fraud has not been reported yet by day 90, so strictly fewer are known.
    assert known_early < known_all


# --------------------------------------------------------------------------- #
# Detector A -- post-outcome field
# --------------------------------------------------------------------------- #
def test_detect_post_outcome_field(tx, row_fraud):
    d = audit.detect_post_outcome_field(tx, row_fraud)
    assert d["verdict"] == "LEAK"
    # Perfect separation: presence of a code lines up exactly with fraud.
    assert d["presence_label_agreement"] == 1.0
    assert d["single_feature_auc"] > 0.99


# --------------------------------------------------------------------------- #
# Detector B -- look-ahead target encoding
# --------------------------------------------------------------------------- #
def test_detect_lookahead_target_encoding(tx, row_fraud):
    d = audit.detect_lookahead_target_encoding(tx, row_fraud)
    assert d["verdict"] == "LEAK"
    # The fingerprint: constant within every merchant.
    assert d["merchants_with_varying_value"] == 0
    assert d["matches_full_history_fraud_mean"] is True


# --------------------------------------------------------------------------- #
# Detector C -- temporal look-ahead window
# --------------------------------------------------------------------------- #
def test_detect_temporal_lookahead(tx):
    d = audit.detect_temporal_lookahead(tx)
    assert d["verdict"] == "LEAK"
    # Matches a FORWARD recompute, not a backward one.
    assert d["match_forward_recompute"] > 0.999
    assert d["match_backward_recompute"] < 0.999


def test_forward_backward_counts_on_toy_example():
    """Hand-checked: three txns for one card, 10h apart, 24h window.

    t0 has two later txns within 24h (t1 at +10h, t2 at +20h) -> forward 2.
    t1 has one later within 24h (t2) and one earlier within 24h (t0) -> fwd 1, bwd 1.
    t2 has two earlier within 24h -> backward 2, forward 0.
    """
    card = np.array(["C1", "C1", "C1"], dtype=object)
    base = np.datetime64("2024-01-01T00:00:00")
    ts = np.array([base, base + np.timedelta64(10, "h"), base + np.timedelta64(20, "h")])
    fwd, bwd = audit._forward_and_backward_counts(card, ts, horizon_h=24)
    assert list(fwd) == [2, 1, 0]
    assert list(bwd) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Univariate screen -- one test is NOT enough
# --------------------------------------------------------------------------- #
def test_univariate_screen_flags_only_the_obvious_leak(tx, row_fraud):
    """The screen catches the post-outcome field but MISSES the look-ahead leaks.

    This is the whole reason we need archetype-specific detectors.
    """
    screen = audit.screen_single_feature_auc(tx, row_fraud)
    flagged = screen["flagged_suspiciously_perfect"]
    assert "has_dispute_code" in flagged
    # The two look-ahead leaks do NOT trip a naive univariate screen.
    assert "merchant_fraud_rate_lifetime" not in flagged
    assert "card_txn_count_next_24h" not in flagged


# --------------------------------------------------------------------------- #
# Temporal split integrity
# --------------------------------------------------------------------------- #
def test_temporal_split_is_past_vs_future(tx):
    split = audit.temporal_split(tx, train_frac=0.70)
    # No row is in both sets; every row is in exactly one.
    assert not (split.train_mask & split.test_mask).any()
    assert (split.train_mask | split.test_mask).all()
    # Train is strictly before the cutoff; test is at/after it.
    assert tx["event_ts"][split.train_mask].max() < split.cutoff
    assert tx["event_ts"][split.test_mask].min() >= split.cutoff
    # Roughly the requested proportion (allow slack for ties at the cutoff).
    assert 0.6 < split.n_train / len(tx) < 0.8


# --------------------------------------------------------------------------- #
# Honest features carry no label information
# --------------------------------------------------------------------------- #
def test_honest_features_are_clean(tx, row_fraud):
    feats = audit.build_honest_features(tx)
    assert len(feats) == len(tx)
    assert not feats.isna().any().any()  # no NaNs to trip the model
    # No honest feature should separate fraud near-perfectly on its own.
    for col in feats.columns:
        x = feats[col].to_numpy(dtype=float)
        if np.unique(x).size < 2:
            continue
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(row_fraud, x)
        assert max(auc, 1 - auc) < 0.95


# --------------------------------------------------------------------------- #
# The headline: the AUC gap
# --------------------------------------------------------------------------- #
def test_leakage_inflates_auc(tx, labels):
    r = audit.run_audit(tx, labels)
    # With the leaks the model is a near-perfect liar.
    assert r.auc_with_leaks > 0.99
    # Without them, an honest, believable score.
    assert 0.70 < r.auc_without_leaks < 0.90
    # The whole lesson: a large, real inflation attributable purely to leakage.
    assert r.auc_gap > 0.10
    # All three detectors returned a LEAK verdict.
    verdicts = [d["verdict"] for d in r.detectors]
    assert verdicts == ["LEAK", "LEAK", "LEAK"]
