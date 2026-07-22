"""Milestone 3 -- the point-in-time-correct feature pipeline.

Milestone 2 proved three features were leaks. This milestone builds the *honest*
replacements: every feature for a transaction at time ``t`` uses ONLY information
that existed strictly before ``t``. Nothing peeks forward. That is what makes an
offline AUC believable in production.

Two leaks from M2 get honest twins here:

    Leak C  card_txn_count_NEXT_24h   (forward)  ->  card_txn_count_prev_24h (backward)
    Leak B  merchant_fraud_rate_LIFETIME (all time) -> merchant_fraud_rate_asof (as-of)

Plus the label itself becomes honest: at a chosen ``as_of`` date, a transaction
counts as fraud only if its chargeback had already been *reported* by then
(``reported_at <= as_of``). Recent frauds that have not come back yet are labelled
0 -- which is exactly the uncomfortable truth a real fraud team trains against.

Design guarantees (each has a test):
  * Backward-only: truncating or deleting future rows does not change any feature
    value for the rows that remain. (`test_features_are_future_proof`)
  * As-of merchant risk is NOT constant within a merchant (the M2 leak was) and
    uses a Bayesian prior so brand-new merchants do not divide by zero.
  * The feature spec (`FEATURE_COLUMNS`) is a single source of truth so training
    and serving compute the identical set -- the seed of "feature-store-lite".

Run it:  ``python -m lonestar.features --data data/raw --out data/features``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lonestar import _labeling

# Bayesian prior for as-of merchant fraud rate. The prior mean alpha/(alpha+beta)
# sits near the ~0.15% base rate, so a merchant with little history is pulled
# toward "typical", not toward 0 or 1.
_PRIOR_ALPHA = 1.0
_PRIOR_BETA = 650.0

# Window lengths (in seconds) for the backward velocity features.
_H = 3600
_WINDOWS = {"prev_1h": _H, "prev_24h": 24 * _H, "prev_7d": 7 * 24 * _H}

# Sentinel for "no prior transaction on this card" (30 days in seconds).
_NO_PRIOR_GAP_S = 30 * 24 * _H

# The single source of truth for which columns the model sees. Training and
# serving both import this, so they can never drift apart.
FEATURE_COLUMNS = (
    # --- row-static (already point-in-time-safe) ---
    "log_amount",
    "hour",
    "dow",
    "is_weekend",
    "entry_CHIP",
    "entry_CONTACTLESS",
    "entry_SWIPE",
    "entry_ECOM",
    "entry_MANUAL",
    "is_foreign",
    "mcc_missing",
    "is_capture",
    # --- card velocity (backward windows only) ---
    "card_txn_count_prev_1h",
    "card_txn_count_prev_24h",
    "card_txn_count_prev_7d",
    "card_amount_sum_prev_24h",
    "secs_since_prev_txn",
    "amount_vs_card_prior_mean",
    # --- merchant risk (as-of, with prior) ---
    "merchant_txns_seen",
    "merchant_fraud_rate_asof",
)

# The two merchant as-of columns are point-in-time-CORRECT but DIAGNOSTIC on this
# dataset: under the 30-60 day label delay a newly-attacking ring merchant looks
# deceptively clean (its fraud has not been reported yet), so
# ``merchant_fraud_rate_asof`` is *inverted* on the drift window. That is a real,
# documented finding -- the label delay blinds you to new attacks precisely when
# they begin -- and it is exactly why this project later builds drift monitoring
# (M8) and retraining (M9). A linear model is misled by the sign flip; the tree
# model in M4 is not. ``ROBUST_FEATURE_COLUMNS`` is the subset that generalises
# under a linear probe, used for the honest-baseline sanity check below.
DIAGNOSTIC_FEATURE_COLUMNS = ("merchant_txns_seen", "merchant_fraud_rate_asof")
ROBUST_FEATURE_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in DIAGNOSTIC_FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Row-static features (unchanged from the M2 honest set -- these never leak)
# --------------------------------------------------------------------------- #
def _row_static(tx: pd.DataFrame) -> pd.DataFrame:
    ts = tx["event_ts"]
    f = pd.DataFrame(index=tx.index)
    f["log_amount"] = np.log1p(tx["amount"].to_numpy())
    f["hour"] = ts.dt.hour.to_numpy()
    f["dow"] = ts.dt.dayofweek.to_numpy()
    f["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int).to_numpy()
    for mode in ("CHIP", "CONTACTLESS", "SWIPE", "ECOM", "MANUAL"):
        f[f"entry_{mode}"] = (tx["entry_mode"] == mode).astype(int).to_numpy()
    f["is_foreign"] = (tx["pos_country"] != "US").astype(int).to_numpy()
    f["mcc_missing"] = tx["mcc"].isna().astype(int).to_numpy()
    f["is_capture"] = (tx["event_type"] == "CAPTURE").astype(int).to_numpy()
    return f


# --------------------------------------------------------------------------- #
# Card velocity -- backward windows only (the honest twin of Leak C)
# --------------------------------------------------------------------------- #
def _card_velocity(tx: pd.DataFrame) -> pd.DataFrame:
    """Per-card counts/sums over PAST windows, computed with searchsorted.

    Everything is strictly backward-looking: for a row at time ``t`` we count only
    events with time in ``[t - window, t)`` on the same card. Ties at the exact
    same timestamp are excluded (side='left'), so a row never counts itself.
    """
    card = tx["card_id"].to_numpy()
    ts = tx["event_ts"].to_numpy().astype("datetime64[s]").astype(np.int64)  # seconds
    amt = tx["amount"].to_numpy(dtype=float)
    n = len(tx)

    # Sort by card, then time, so each card is a contiguous, time-ordered block.
    order = np.lexsort((ts, card))
    card_s, ts_s, amt_s = card[order], ts[order], amt[order]

    out = {
        "card_txn_count_prev_1h": np.zeros(n, dtype=np.int64),
        "card_txn_count_prev_24h": np.zeros(n, dtype=np.int64),
        "card_txn_count_prev_7d": np.zeros(n, dtype=np.int64),
        "card_amount_sum_prev_24h": np.zeros(n, dtype=float),
        "secs_since_prev_txn": np.full(n, _NO_PRIOR_GAP_S, dtype=float),
        "amount_vs_card_prior_mean": np.ones(n, dtype=float),
    }

    change = np.flatnonzero(card_s[1:] != card_s[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))

    for s, e in zip(starts, ends, strict=True):
        t = ts_s[s:e]
        a = amt_s[s:e]
        # Upper bound: first index at-or-after t (strictly-before count).
        upto = np.searchsorted(t, t, side="left")
        # Counts over each backward window.
        for name, w in _WINDOWS.items():
            lo = np.searchsorted(t, t - w, side="left")
            out[f"card_txn_count_{name}"][s:e] = upto - lo
        # Amount sum over the 24h window via a prefix-sum array.
        prefix = np.concatenate(([0.0], np.cumsum(a)))
        lo24 = np.searchsorted(t, t - _WINDOWS["prev_24h"], side="left")
        out["card_amount_sum_prev_24h"][s:e] = prefix[upto] - prefix[lo24]
        # Seconds since the immediately previous transaction on this card.
        gaps = np.diff(t)
        out["secs_since_prev_txn"][s + 1 : e] = gaps
        # Current amount vs the card's mean of ALL prior amounts (expanding mean).
        prior_count = upto  # number of strictly-earlier txns
        prior_mean = np.divide(
            prefix[upto], prior_count, out=np.zeros_like(a), where=prior_count > 0
        )
        # Where there is no prior, ratio is 1.0 (neutral).
        ratio = np.where(prior_count > 0, a / (prior_mean + 1.0), 1.0)
        out["amount_vs_card_prior_mean"][s:e] = ratio

    # Undo the sort so rows line up with the original frame.
    result = pd.DataFrame(index=tx.index)
    for name, arr in out.items():
        restored = np.empty(n, dtype=arr.dtype)
        restored[order] = arr
        result[name] = restored
    return result


# --------------------------------------------------------------------------- #
# Merchant risk -- as-of, with a prior (the honest twin of Leak B)
# --------------------------------------------------------------------------- #
def _merchant_asof_risk(tx: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """For each row, the merchant's fraud rate KNOWN at that instant.

    Numerator: chargebacks on this merchant already *reported* by ``t``.
    Denominator: this merchant's transactions seen strictly before ``t``.
    Smoothed by a Beta(alpha, beta) prior so a fresh merchant is neutral, not 0/0.

    Unlike the M2 leak, this value RISES over time as history and chargebacks
    accrue -- so it varies within a merchant, which is the tell that it is honest.
    """
    # Map every chargeback to its merchant via the AUTH row's merchant_id.
    auth = (
        tx[tx["event_type"] == "AUTH"][["auth_code", "merchant_id"]]
        .drop_duplicates("auth_code")
        .set_index("auth_code")["merchant_id"]
    )
    lab = labels.copy()
    lab["merchant_id"] = lab["auth_code"].map(auth)
    reported_by_merchant = {
        m: np.sort(g["reported_at"].to_numpy().astype("datetime64[s]").astype(np.int64))
        for m, g in lab.groupby("merchant_id")
    }

    merch = tx["merchant_id"].to_numpy()
    ts = tx["event_ts"].to_numpy().astype("datetime64[s]").astype(np.int64)
    n = len(tx)
    seen = np.zeros(n, dtype=np.int64)
    rate = np.zeros(n, dtype=float)

    order = np.lexsort((ts, merch))
    merch_s, ts_s = merch[order], ts[order]
    change = np.flatnonzero(merch_s[1:] != merch_s[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))

    for s, e in zip(starts, ends, strict=True):
        m = merch_s[s]
        t = ts_s[s:e]
        # Transactions on this merchant strictly before each t.
        denom = np.searchsorted(t, t, side="left")
        # Chargebacks on this merchant reported at-or-before each t.
        rep = reported_by_merchant.get(m)
        numer = np.searchsorted(rep, t, side="right") if rep is not None else np.zeros_like(denom)
        seen[s:e] = denom
        rate[s:e] = (numer + _PRIOR_ALPHA) / (denom + _PRIOR_ALPHA + _PRIOR_BETA)

    result = pd.DataFrame(index=tx.index)
    seen_r = np.empty(n, dtype=np.int64)
    rate_r = np.empty(n, dtype=float)
    seen_r[order] = seen
    rate_r[order] = rate
    result["merchant_txns_seen"] = seen_r
    result["merchant_fraud_rate_asof"] = rate_r
    return result


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def compute_features(tx: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Full point-in-time feature matrix, columns in ``FEATURE_COLUMNS`` order."""
    feats = pd.concat(
        [_row_static(tx), _card_velocity(tx), _merchant_asof_risk(tx, labels)], axis=1
    )
    # Enforce the spec: exact columns, exact order. Guards train/serve parity.
    return feats[list(FEATURE_COLUMNS)]


def build_training_frame(
    tx: pd.DataFrame, labels: pd.DataFrame, as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Features + honest (censored) label + keys, for rows known by ``as_of``.

    If ``as_of`` is given: keep only transactions that had happened by then, and
    label a row fraud only if its chargeback was reported by then. If ``None``,
    use the full data and eventual labels (useful for offline analysis).
    """
    frame = tx
    if as_of is not None:
        frame = tx[tx["event_ts"] <= as_of].copy()

    feats = compute_features(frame, labels)
    y = _labeling.attach_row_label(frame, labels, as_of=as_of).to_numpy()

    out = feats.copy()
    out.insert(0, "transaction_id", frame["transaction_id"].to_numpy())
    out.insert(1, "event_ts", frame["event_ts"].to_numpy())
    out["is_fraud"] = y.astype(int)
    return out


# --------------------------------------------------------------------------- #
# Validation helpers (the milestone's real guarantees)
# --------------------------------------------------------------------------- #
def is_future_proof(tx: pd.DataFrame, labels: pd.DataFrame, quantile: float = 0.80) -> bool:
    """THE guarantee: deleting future rows must not change any surviving feature.

    We compute features on the full data, then recompute on data truncated to the
    earliest ``quantile`` of time, and check the surviving rows are byte-for-byte
    identical. If any feature peeked forward, the truncated values would differ.
    """
    full = compute_features(tx, labels)
    cutoff = tx["event_ts"].quantile(quantile)
    early = tx[tx["event_ts"] <= cutoff]
    early_feats = compute_features(early, labels)
    return bool(np.allclose(early_feats.to_numpy(), full.loc[early.index].to_numpy()))


def temporal_split_auc(
    tx: pd.DataFrame, labels: pd.DataFrame, cols: tuple | None = None, train_frac: float = 0.70
) -> float:
    """Honest past->future AUC from a linear probe over the given feature columns.

    A sanity check that the pipeline carries real signal -- NOT the project's final
    model (that is Milestone 4). Defaults to the robust subset because a linear
    model is misled by the diagnostic merchant-rate inversion (see the note above).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cols = list(cols) if cols is not None else list(ROBUST_FEATURE_COLUMNS)
    feats = compute_features(tx, labels)[cols]
    y = _labeling.attach_row_label(tx, labels).to_numpy()
    cutoff = tx["event_ts"].quantile(train_frac)
    train = (tx["event_ts"] < cutoff).to_numpy()
    test = (tx["event_ts"] >= cutoff).to_numpy()

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(feats[train], y[train])
    proba = model.predict_proba(feats[test])[:, 1]
    return float(roc_auc_score(y[test], proba))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the point-in-time feature table.")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="data/features")
    ap.add_argument(
        "--as-of",
        default=None,
        help="ISO date; label/censor as of this instant. Default: full data, eventual labels.",
    )
    args = ap.parse_args()

    data = Path(args.data)
    tx = pd.read_parquet(data / "transactions.parquet")
    labels = pd.read_parquet(data / "chargeback_labels.parquet")

    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    frame = build_training_frame(tx, labels, as_of=as_of)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "features.parquet", index=False)

    # Honest summary.
    rate = frame["is_fraud"].mean()
    print(f"rows: {len(frame):,}")
    print(f"features: {len(FEATURE_COLUMNS)}")
    print(f"as_of: {as_of}")
    print(f"labelled fraud rate (censored): {rate:.4%}")
    print(f"wrote {out / 'features.parquet'}")

    # The milestone's guarantees, checked on the fly.
    print("\n--- point-in-time guarantees ---")
    print(f"future-proof (no feature uses future data): {is_future_proof(tx, labels)}")
    robust = temporal_split_auc(tx, labels, cols=ROBUST_FEATURE_COLUMNS)
    diag = temporal_split_auc(tx, labels, cols=DIAGNOSTIC_FEATURE_COLUMNS)
    print(f"honest temporal-split AUC (robust features): {robust:.4f}")
    print(
        f"diagnostic merchant as-of AUC: {diag:.4f}  "
        "(<0.5 = label delay hides new attackers -> motivates M8/M9)"
    )


if __name__ == "__main__":
    main()
