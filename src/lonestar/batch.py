"""Milestone 7 -- the batch scorer.

Scores a whole table of transactions at once (a nightly job, a backfill, a
what-if run) and writes decisions to disk. It is the offline twin of the online
service from Milestone 6, and -- crucially -- it makes its decisions through the
SAME shared core (``lonestar.scoring``) and the SAME model + policy artifacts.

That shared path is what guarantees **train/serve parity**: the online API and the
batch job cannot disagree about a transaction, because they run identical segment
recovery, threshold lookup, and decision logic over an identical model. The
proof is ``test_online_batch_parity`` in the test suite.

Feature parity is structural too: both training and scoring build features from
the one definition in ``lonestar.features`` (``FEATURE_COLUMNS``), so the columns
can never silently drift apart.

Run it:  ``python -m lonestar.batch --features data/features/features.parquet \
             --model models/fraud_model.joblib --policy models/decision_policy.json \
             --out data/scored/scored.parquet``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lonestar import scoring
from lonestar.features import FEATURE_COLUMNS


def _recover_segments(df: pd.DataFrame) -> np.ndarray:
    """Vectorised segment recovery -- identical rule to scoring.recover_segment
    (the first entry one-hot that is set wins), applied to the whole frame."""
    entry_cols = list(scoring.ENTRY_ONEHOTS)
    names = np.array([scoring.ENTRY_ONEHOTS[c] for c in entry_cols], dtype=object)
    mask = df[entry_cols].to_numpy() >= 0.5  # (n, 5) booleans
    has_any = mask.any(axis=1)
    first_idx = mask.argmax(axis=1)  # argmax on bool -> index of first True
    return np.where(has_any, names[first_idx], None)


def score_batch(df: pd.DataFrame, model, policy: dict | None) -> pd.DataFrame:
    """Score every row: probability, segment, threshold, decision.

    ``df`` must contain ``FEATURE_COLUMNS`` (and ``transaction_id`` if present).
    Uses vectorised ``predict_proba`` for speed, then the SHARED decision core so
    the result matches the online service exactly.
    """
    proba = model.predict_proba(df[list(FEATURE_COLUMNS)])[:, 1]
    segments = _recover_segments(df)

    # Map each distinct segment to its threshold once, then broadcast.
    seg_series = pd.Series(segments, index=df.index)
    seg_to_threshold = {s: scoring.threshold_for(policy, s) for s in set(segments)}
    thresholds = seg_series.map(seg_to_threshold).to_numpy(dtype=float)

    decisions = np.where(proba >= thresholds, "DECLINE", "APPROVE")

    out = pd.DataFrame(
        {
            "fraud_probability": np.round(proba, 6),
            "segment": segments,
            "threshold": np.round(thresholds, 4),
            "decision": decisions,
        },
        index=df.index,
    )
    if "transaction_id" in df.columns:
        out.insert(0, "transaction_id", df["transaction_id"].to_numpy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-score a feature table (M7).")
    ap.add_argument("--features", default="data/features/features.parquet")
    ap.add_argument("--model", default="models/fraud_model.joblib")
    ap.add_argument("--policy", default="models/decision_policy.json")
    ap.add_argument("--out", default="data/scored/scored.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    model = joblib.load(args.model)
    policy_path = Path(args.policy)
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else None

    scored = score_batch(df, model, policy)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out, index=False)

    n = len(scored)
    declines = int((scored["decision"] == "DECLINE").sum())
    print(f"scored: {n:,} transactions")
    print(f"declines: {declines:,} ({declines / n:.3%})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
