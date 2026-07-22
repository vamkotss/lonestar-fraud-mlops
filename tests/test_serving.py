"""Tests for the Milestone 6 online scoring service.

A module fixture builds a real model + decision policy in a temp dir, then the
FastAPI TestClient exercises the endpoint: auth, schema safety, decision
correctness against the policy, reason codes, and the p99 latency budget.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lonestar import decisions, features, modeling, serving

_API_KEY = "test-key-123"


@pytest.fixture(scope="module")
def model_dir(tx, labels, tmp_path_factory):
    """Train a real model and fit a policy into a temp model dir."""
    d = tmp_path_factory.mktemp("model")
    frame = features.build_training_frame(tx, labels)
    X = frame[list(features.FEATURE_COLUMNS)]
    y = frame["is_fraud"].to_numpy()
    ts = frame["event_ts"]
    model, metrics, split = modeling.train_final(X, y, ts, "xgb")
    modeling.save_model(model, "xgb", d, metrics, split, 0.005)

    # Fit a decision policy on the scored frame.
    scored = frame.merge(
        tx[["transaction_id", "amount", "entry_mode"]], on="transaction_id", how="left"
    )
    scored["proba"] = model.predict_proba(scored[list(features.FEATURE_COLUMNS)])[:, 1]
    policy = decisions.fit_segment_policy(scored, decisions.CostModel())
    (d / "decision_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return d


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("LS_API_KEY", _API_KEY)
    app = serving.create_app(model_dir)
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_features(tx, labels) -> dict:
    """A real feature row as a name->value dict the API expects."""
    frame = features.build_training_frame(tx, labels)
    row = frame[list(features.FEATURE_COLUMNS)].iloc[0]
    return {c: float(row[c]) for c in features.FEATURE_COLUMNS}


def _auth():
    return {"Authorization": f"Bearer {_API_KEY}"}


# --------------------------------------------------------------------------- #
# Health + auth
# --------------------------------------------------------------------------- #
def test_health_needs_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_requires_auth(client, sample_features):
    body = {"transaction_id": "t1", "features": sample_features}
    # No header at all.
    assert client.post("/score", json=body).status_code == 401
    # Wrong key.
    r = client.post("/score", json=body, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


# --------------------------------------------------------------------------- #
# Schema safety
# --------------------------------------------------------------------------- #
def test_missing_feature_is_rejected(client, sample_features):
    bad = dict(sample_features)
    bad.pop("log_amount")  # drop a required feature
    r = client.post("/score", json={"transaction_id": "t2", "features": bad}, headers=_auth())
    assert r.status_code == 422
    assert "log_amount" in r.json()["detail"]["missing_features"]


def test_unknown_feature_is_rejected(client, sample_features):
    bad = dict(sample_features)
    bad["not_a_real_feature"] = 1.0
    r = client.post("/score", json={"transaction_id": "t3", "features": bad}, headers=_auth())
    assert r.status_code == 422
    assert "not_a_real_feature" in r.json()["detail"]["unknown_features"]


# --------------------------------------------------------------------------- #
# A valid score: shape, reason codes, decision matches the policy
# --------------------------------------------------------------------------- #
def test_valid_score_shape_and_reasons(client, sample_features):
    r = client.post(
        "/score", json={"transaction_id": "t4", "features": sample_features}, headers=_auth()
    )
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "t4"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in ("APPROVE", "DECLINE")
    assert len(body["reason_codes"]) == 5
    for rc in body["reason_codes"]:
        assert rc["direction"] in ("raises", "lowers")
    assert body["latency_ms"] >= 0


def test_decision_matches_policy(client, model_dir, sample_features):
    policy = json.loads((model_dir / "decision_policy.json").read_text())
    r = client.post(
        "/score", json={"transaction_id": "t5", "features": sample_features}, headers=_auth()
    )
    body = r.json()
    # Recover the segment the app used and its threshold, then re-derive the call.
    seg = body["segment"]
    expected_threshold = policy["segment_thresholds"].get(seg, policy["global_threshold"])
    assert abs(body["threshold"] - round(expected_threshold, 4)) < 1e-6
    expected_decision = "DECLINE" if body["fraud_probability"] >= body["threshold"] else "APPROVE"
    assert body["decision"] == expected_decision


# --------------------------------------------------------------------------- #
# Latency budget (p99 < 100ms)
# --------------------------------------------------------------------------- #
def test_latency_p99_under_budget(client, sample_features):
    body = {"transaction_id": "lat", "features": sample_features}
    # Warm up (first call triggers lazy artifact load).
    for _ in range(5):
        client.post("/score", json=body, headers=_auth())
    server_latencies = []
    for _ in range(100):
        r = client.post("/score", json=body, headers=_auth())
        server_latencies.append(r.json()["latency_ms"])
    p99 = float(np.percentile(server_latencies, 99))
    assert p99 < 100.0, f"p99 {p99:.1f}ms exceeds 100ms budget"


# --------------------------------------------------------------------------- #
# Model-agnostic: the service also serves a linear (LR pipeline) winner
# --------------------------------------------------------------------------- #
@pytest.fixture()
def lr_model_dir(tx, labels, tmp_path_factory):
    """Save a LinearRegression-style winner (scaler + LogisticRegression pipeline)."""
    d = tmp_path_factory.mktemp("lr_model")
    frame = features.build_training_frame(tx, labels)
    X = frame[list(features.FEATURE_COLUMNS)]
    y = frame["is_fraud"].to_numpy()
    ts = frame["event_ts"]
    model, metrics, split = modeling.train_final(X, y, ts, "lr")  # a Pipeline
    modeling.save_model(model, "lr", d, metrics, split, 0.005)
    scored = frame.merge(
        tx[["transaction_id", "amount", "entry_mode"]], on="transaction_id", how="left"
    )
    scored["proba"] = model.predict_proba(scored[list(features.FEATURE_COLUMNS)])[:, 1]
    policy = decisions.fit_segment_policy(scored, decisions.CostModel())
    (d / "decision_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return d


def test_serves_a_linear_pipeline_winner(lr_model_dir, sample_features, monkeypatch):
    """Regression guard: a scaler+LogisticRegression pipeline must score AND
    still return reason codes (via coefficient contributions, not TreeSHAP)."""
    monkeypatch.setenv("LS_API_KEY", _API_KEY)
    client = TestClient(serving.create_app(lr_model_dir))
    r = client.post(
        "/score", json={"transaction_id": "lr1", "features": sample_features}, headers=_auth()
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in ("APPROVE", "DECLINE")
    assert len(body["reason_codes"]) == 5  # linear contributions still produced
