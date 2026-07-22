"""Tests for the Milestone 2 EDA computations.

Fast, figure-free checks (matplotlib rendering is exercised by the CLI, not here)
that the headline findings match the physics the generator guarantees.
"""

from __future__ import annotations

from lonestar import eda


def test_overall_fraud_rate_is_rare(tx, labels):
    f = eda.compute_eda(tx, labels)
    # A fraction of a percent -- the imbalance that defines the project.
    assert 0.0010 < f["overall_fraud_rate"] < 0.0060


def test_ring_lifts_the_fraud_rate(tx, labels):
    f = eda.compute_eda(tx, labels)
    # Post-onset fraud is materially higher than pre-onset.
    assert f["post_onset_fraud_rate"] > f["pre_onset_fraud_rate"] * 1.5


def test_ring_shifts_the_feature_mix(tx, labels):
    f = eda.compute_eda(tx, labels)
    rs = f["ring_signature"]
    # The ring routes fraud toward e-commerce and foreign geographies.
    assert rs["ecom_share_fraud_post"] > rs["ecom_share_fraud_pre"]
    assert rs["foreign_share_fraud_post"] > rs["foreign_share_fraud_pre"]


def test_data_quality_flags_present(tx, labels):
    f = eda.compute_eda(tx, labels)
    dq = f["data_quality"]
    assert 0.03 < dq["missing_mcc_rate"] < 0.05
    assert dq["names_per_merchant"] > 1.0  # chaotic descriptors
    assert dq["tz_naive"] is True
