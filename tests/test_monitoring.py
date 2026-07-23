"""Tests for the Milestone 8 drift monitor.

The headline tests are:
  * ``test_detects_ring_without_labels`` -- the monitor alerts at the ring onset.
  * ``test_population_psi_misses_what_segment_monitoring_catches`` -- the finding
    that motivates the whole design.
  * ``test_monitoring_path_is_label_free`` -- nothing in the alerting path reads
    the fraud label, because in production it would not exist yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lonestar import decisions, features, modeling, monitoring

_RING_ONSET_MONTH = 13  # first month the ring's effect is visible in the data


@pytest.fixture(scope="module")
def scored(tx, labels) -> pd.DataFrame:
    """Feature frame + model scores + segment -- what the monitor consumes."""
    frame = features.build_training_frame(tx, labels)
    X = frame[list(features.FEATURE_COLUMNS)]
    y = frame["is_fraud"].to_numpy()
    ts = frame["event_ts"]
    model, _, _ = modeling.train_final(X, y, ts, "xgb")
    out = frame.merge(
        tx[["transaction_id", "amount", "entry_mode"]], on="transaction_id", how="left"
    )
    out["proba"] = model.predict_proba(out[list(features.FEATURE_COLUMNS)])[:, 1]
    return out


# --------------------------------------------------------------------------- #
# PSI mechanics
# --------------------------------------------------------------------------- #
def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert monitoring.compute_psi(x, x) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    small = monitoring.compute_psi(ref, rng.normal(0.2, 1, 5000))
    large = monitoring.compute_psi(ref, rng.normal(2.0, 1, 5000))
    assert small < large
    assert large >= monitoring.PSI_MAJOR  # a 2-sigma shift is a major move


def test_psi_handles_constant_feature():
    """A zero-variance reference must not divide by zero or blow up."""
    ref = np.zeros(1000)
    assert monitoring.compute_psi(ref, np.zeros(1000)) < 1e-6
    assert monitoring.compute_psi(ref, np.ones(1000)) > 0


def test_psi_bands():
    assert monitoring.psi_band(0.05) == "stable"
    assert monitoring.psi_band(0.15) == "moderate"
    assert monitoring.psi_band(0.40) == "major"


# --------------------------------------------------------------------------- #
# Alert rate + baseline
# --------------------------------------------------------------------------- #
def test_alert_rate_counts_the_tail():
    scores = np.array([0.1, 0.2, 0.6, 0.9])
    assert monitoring.alert_rate(scores, 0.5) == 0.5
    assert monitoring.alert_rate(np.array([]), 0.5) == 0.0


def test_tail_threshold_is_a_baseline_quantile():
    scores = np.linspace(0, 1, 1001)
    thr = monitoring.tail_threshold(scores, quantile=0.99)
    assert 0.985 < thr < 0.995
    # By construction ~1% of the baseline sits at or above its own 99th percentile.
    assert abs(monitoring.alert_rate(scores, thr) - 0.01) < 0.005


def test_baseline_stats_per_segment(scored):
    ts = scored["event_ts"]
    baseline = scored[ts < ts.quantile(monitoring.BASELINE_FRAC)]
    stats = monitoring.baseline_stats(baseline)
    assert "__overall__" in stats
    for seg in scored["entry_mode"].unique():
        st = stats[str(seg)]
        assert st["n_months"] > 1  # enough history for a control limit
        # The tail rate is pinned near (1 - quantile) BY CONSTRUCTION, which is what
        # makes the monitor independent of how the model happens to be calibrated.
        assert abs(st["mean"] - (1 - monitoring.TAIL_QUANTILE)) < 0.005


# --------------------------------------------------------------------------- #
# THE headline: the monitor catches the ring, label-free
# --------------------------------------------------------------------------- #
def test_detects_ring_without_labels(scored):
    report = monitoring.monitor(scored)
    first = report["first_alert_month"]
    assert first is not None, "monitor never alerted on a dataset containing a fraud ring"
    # It fires at (or within a month of) the ring switching on -- long before the
    # 30-60 day chargeback delay would reveal anything.
    assert first <= _RING_ONSET_MONTH + 1


def test_ecommerce_is_the_flagged_segment(scored):
    """The ring attacks e-commerce, so ECOM should be the worst-drifting segment."""
    report = monitoring.monitor(scored)
    alerting = [w for w in report["windows"] if w["status"] in ("WARN", "ALERT")]
    assert alerting, "expected at least one WARN/ALERT window"
    worst_segments = {w["segments"][0]["segment"] for w in alerting if w["segments"]}
    assert "ECOM" in worst_segments


def test_population_psi_misses_what_segment_monitoring_catches(scored):
    """The finding: averaged drift metrics wash out a rare subpopulation.

    Population score PSI stays in the stable/moderate band throughout, while the
    segment tail alert rate goes many sigma above baseline. This is why the alert
    rule is built on segment tails rather than population PSI.
    """
    report = monitoring.monitor(scored)
    # Population PSI never reaches the "major" band...
    assert not report["population_psi_ever_major"]
    # ...but some segment goes well past the ALERT control limit.
    max_z = max(
        (s["zscore"] or 0) for w in report["windows"] for s in w["segments"]
    )
    assert max_z >= monitoring.SIGMA_ALERT


def test_monitoring_path_is_label_free(scored):
    """Drop the label entirely; the monitor must still produce the same alerts.

    In production, chargebacks for the current month have not arrived. If the
    monitor needed labels it would be useless exactly when it matters most.
    """
    with_label = monitoring.monitor(scored)
    unlabelled = scored.drop(columns=["is_fraud"])
    without_label = monitoring.monitor(unlabelled)
    assert without_label["first_alert_month"] == with_label["first_alert_month"]
    assert [w["status"] for w in without_label["windows"]] == [
        w["status"] for w in with_label["windows"]
    ]


def test_alerts_are_independent_of_model_calibration(scored):
    """Regression guard for a bug found at full scale.

    A fixed score cutoff (e.g. 0.5) is a genuine tail for one model and captures
    30% of the population for a more aggressively calibrated one -- at which point
    the "tail rate" measures the bulk, baseline variance explodes, and the monitor
    never fires. Because the tail is a quantile of each segment's OWN baseline,
    any monotonic re-calibration of the scores must leave the alerts unchanged.
    """
    original = monitoring.monitor(scored)

    # A monotonic squash that massively inflates how many rows sit above 0.5,
    # exactly like the imbalance-weighted full-scale model did.
    recalibrated = scored.copy()
    recalibrated["proba"] = recalibrated["proba"] ** 0.25

    shifted = monitoring.monitor(recalibrated)
    assert shifted["first_alert_month"] == original["first_alert_month"]
    assert [w["status"] for w in shifted["windows"]] == [
        w["status"] for w in original["windows"]
    ]


def test_tiny_windows_are_skipped(scored):
    """A truncated final month must not fire a false alarm on calendar features."""
    report = monitoring.monitor(scored)
    for w in report["windows"]:
        assert w["n_rows"] >= monitoring.MIN_WINDOW_ROWS


# --------------------------------------------------------------------------- #
# Report shape
# --------------------------------------------------------------------------- #
def test_report_contains_baseline_and_windows(scored):
    report = monitoring.monitor(scored)
    assert report["baseline"]["n_rows"] > 0
    assert report["windows"]
    for w in report["windows"]:
        assert w["status"] in ("OK", "WARN", "ALERT")
        assert 0.0 <= w["overall_alert_rate"] <= 1.0


def test_markdown_renders(scored):
    report = monitoring.monitor(scored)
    md = monitoring._render_markdown(report)
    assert "# Drift Monitoring Report" in md
    assert "Segment tail alert rate" in md


def test_decisions_score_frame_reused(scored):
    """Sanity: the monitor consumes exactly the frame the decision layer builds."""
    assert "proba" in scored.columns
    assert "entry_mode" in scored.columns
    assert decisions._SEGMENT_COL == "entry_mode"
