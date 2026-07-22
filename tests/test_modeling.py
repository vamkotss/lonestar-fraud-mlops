"""Tests for the Milestone 4 modeling pipeline.

Prove the parts a reviewer cares about:
  * temporal CV never trains on the future (no leakage in the evaluation itself),
  * PR-AUC is computed and sane at this imbalance,
  * XGBoost beats the linear baseline on the honest metric,
  * the winning model is saved with a serving-ready card and reloads correctly.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from lonestar import features, modeling


@pytest.fixture(scope="module")
def frame(tx, labels) -> pd.DataFrame:
    """The M3 feature table (built once for this module)."""
    return features.build_training_frame(tx, labels)


@pytest.fixture(scope="module")
def features_path(frame, tmp_path_factory):
    """Write the feature frame to a temp parquet for the CLI-style ``run``."""
    p = tmp_path_factory.mktemp("feat") / "features.parquet"
    frame.to_parquet(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# Temporal CV must not leak
# --------------------------------------------------------------------------- #
def test_temporal_folds_never_train_on_the_future(frame):
    ts = frame["event_ts"]
    for train, val in modeling.temporal_folds(ts, n_splits=4):
        if not train.any() or not val.any():
            continue
        # Every training row is strictly earlier than every validation row.
        assert ts[train].max() < ts[val].min()


def test_pos_weight_is_neg_over_pos():
    y = np.array([0, 0, 0, 1])
    assert modeling._pos_weight(y) == 3.0


def test_build_model_kinds():
    from sklearn.pipeline import Pipeline

    assert isinstance(modeling.build_model("lr"), Pipeline)
    # xgb import lives inside build_model; just confirm it constructs.
    assert modeling.build_model("xgb", scale_pos_weight=10.0) is not None
    with pytest.raises(ValueError):
        modeling.build_model("nope")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_evaluate_returns_both_metrics():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = modeling.evaluate(y, proba)
    assert m["roc_auc"] == 1.0  # perfectly separated
    assert 0.0 <= m["pr_auc"] <= 1.0


# --------------------------------------------------------------------------- #
# The headline: XGBoost beats the linear baseline on PR-AUC
# --------------------------------------------------------------------------- #
def test_xgboost_beats_linear_baseline(frame):
    X = frame[list(features.FEATURE_COLUMNS)]
    y = frame["is_fraud"].to_numpy()
    ts = frame["event_ts"]
    _, lr_metrics, _ = modeling.train_final(X, y, ts, "lr")
    _, xgb_metrics, _ = modeling.train_final(X, y, ts, "xgb")
    # XGBoost wins on the honest metric, and comfortably on ROC too.
    assert xgb_metrics["pr_auc"] > lr_metrics["pr_auc"]
    assert xgb_metrics["roc_auc"] > 0.75


# --------------------------------------------------------------------------- #
# End-to-end run: selection, save, reload
# --------------------------------------------------------------------------- #
def test_run_selects_and_saves_winner(features_path, tmp_path):
    out = tmp_path / "models"
    report = modeling.run(features_path, out, track=False, n_splits=4)

    # XGBoost should win on PR-AUC, with a large lift over prevalence.
    assert report.winner == "xgb"
    assert report.pr_auc_lift_over_prevalence > 20

    # A serving-ready model + card were written.
    card_path = out / "model_card.json"
    model_path = out / "fraud_model.joblib"
    assert card_path.exists() and model_path.exists()

    import json

    card = json.loads(card_path.read_text())
    # The card records the EXACT feature order the model expects.
    assert card["feature_columns"] == list(features.FEATURE_COLUMNS)
    assert card["primary_metric"] == "pr_auc"

    # The saved model reloads and produces valid probabilities.
    model = joblib.load(model_path)
    df = pd.read_parquet(features_path).head(200)
    proba = model.predict_proba(df[list(features.FEATURE_COLUMNS)])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
