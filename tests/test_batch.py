"""Tests for the Milestone 7 batch scorer, including the online/batch parity test.

The headline is ``test_online_batch_parity``: the same transactions scored through
the live API (M6) and through the batch job (M7) must get identical probabilities
and identical decisions. That is the guarantee against serving skew.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lonestar import batch, decisions, features, modeling, scoring, serving

_API_KEY = "parity-key"


@pytest.fixture(scope="module")
def model_dir(tx, labels, tmp_path_factory):
    """Train a model + fit a policy into a temp dir (shared by these tests)."""
    d = tmp_path_factory.mktemp("m7model")
    frame = features.build_training_frame(tx, labels)
    X = frame[list(features.FEATURE_COLUMNS)]
    y = frame["is_fraud"].to_numpy()
    ts = frame["event_ts"]
    model, metrics, split = modeling.train_final(X, y, ts, "xgb")
    modeling.save_model(model, "xgb", d, metrics, split, 0.005)
    scored = frame.merge(
        tx[["transaction_id", "amount", "entry_mode"]], on="transaction_id", how="left"
    )
    scored["proba"] = model.predict_proba(scored[list(features.FEATURE_COLUMNS)])[:, 1]
    policy = decisions.fit_segment_policy(scored, decisions.CostModel())
    (d / "decision_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def feature_frame(tx, labels) -> pd.DataFrame:
    return features.build_training_frame(tx, labels)


# --------------------------------------------------------------------------- #
# Batch scorer basics
# --------------------------------------------------------------------------- #
def test_score_batch_shape_and_columns(model_dir, feature_frame):
    import joblib

    model = joblib.load(model_dir / "fraud_model.joblib")
    policy = json.loads((model_dir / "decision_policy.json").read_text())
    out = batch.score_batch(feature_frame, model, policy)

    assert len(out) == len(feature_frame)
    assert {"transaction_id", "fraud_probability", "segment", "threshold", "decision"} <= set(
        out.columns
    )
    assert out["decision"].isin(["APPROVE", "DECLINE"]).all()
    assert out["fraud_probability"].between(0, 1).all()


def test_vectorised_segment_recovery_matches_scalar(feature_frame):
    """The batch's vectorised segment recovery must match the scalar core row-by-row."""
    vec = batch._recover_segments(feature_frame.head(200))
    for i, (_, row) in enumerate(feature_frame.head(200).iterrows()):
        scalar = scoring.recover_segment({c: row[c] for c in features.FEATURE_COLUMNS})
        assert vec[i] == scalar


def test_batch_threshold_matches_policy(model_dir, feature_frame):
    import joblib

    model = joblib.load(model_dir / "fraud_model.joblib")
    policy = json.loads((model_dir / "decision_policy.json").read_text())
    out = batch.score_batch(feature_frame.head(500), model, policy)
    for _, r in out.iterrows():
        expected = scoring.threshold_for(policy, r["segment"])
        assert abs(r["threshold"] - round(expected, 4)) < 1e-6


# --------------------------------------------------------------------------- #
# THE parity test: online API == batch job, exactly
# --------------------------------------------------------------------------- #
def test_online_batch_parity(model_dir, feature_frame, monkeypatch):
    """Same rows, two paths (live API vs batch), identical proba and decision."""
    import joblib

    monkeypatch.setenv("LS_API_KEY", _API_KEY)
    client = TestClient(serving.create_app(model_dir))
    model = joblib.load(model_dir / "fraud_model.joblib")
    policy = json.loads((model_dir / "decision_policy.json").read_text())

    # A spread of rows (different segments, some rare high-risk ones).
    sample = feature_frame.sample(n=40, random_state=7)
    batch_out = batch.score_batch(sample, model, policy).set_index("transaction_id")

    headers = {"Authorization": f"Bearer {_API_KEY}"}
    for _, row in sample.iterrows():
        payload = {
            "transaction_id": str(row["transaction_id"]),
            "features": {c: float(row[c]) for c in features.FEATURE_COLUMNS},
        }
        resp = client.post("/score", json=payload, headers=headers).json()
        b = batch_out.loc[row["transaction_id"]]
        # Probabilities agree to within rounding, and the decision is identical.
        assert abs(resp["fraud_probability"] - float(b["fraud_probability"])) < 1e-6
        assert resp["decision"] == b["decision"]
        assert resp["segment"] == (None if b["segment"] is None else b["segment"])
