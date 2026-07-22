"""Milestone 2 -- the leakage audit.

The generator (Milestone 1) planted three data-leakage traps in ``transactions``.
This module is the auditor that *hunts them down and proves each is a leak*, then
quantifies the damage by training the same simple model twice -- once WITH the
leaks and once WITHOUT -- on a strict temporal split.

The single most important idea in the whole project lives here:

    You cannot random-split fraud data.

    A fraud label (chargeback) arrives 30-60 days AFTER the transaction. If you
    shuffle rows and split randomly, "future" rows leak into the training set and
    your test score becomes a fantasy. So we split by TIME: train on the earlier
    window, test on the later window. That is what ``temporal_split`` does, and it
    is why the honest test AUC below is believable while the leaky one is a lie.

Three leaks, three DIFFERENT failure modes -- and, importantly, three different
detectors, because no single test catches all three:

    A. ``dispute_reason_code``      post-outcome field   -> perfect separation
    B. ``merchant_fraud_rate_lifetime``  look-ahead target encoding -> constant per group
    C. ``card_txn_count_next_24h``  temporal look-ahead  -> matches a FORWARD recompute

Run it:  ``python -m lonestar.audit --data data/raw``   (respects LS_SEED)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lonestar import _labeling

# The three planted leak columns, by name. (Also recorded in the generation
# manifest under ``leak_columns`` -- this is the audit's independent copy.)
LEAK_COLUMNS = (
    "dispute_reason_code",
    "merchant_fraud_rate_lifetime",
    "card_txn_count_next_24h",
)

# Any single column that predicts fraud this well on its own is almost certainly
# a leak -- honest signals are never this clean on a 0.15%-positive problem.
_PERFECT_AUC_FLAG = 0.95


# --------------------------------------------------------------------------- #
# Honest features: what we legitimately know AT the moment of the transaction.
# --------------------------------------------------------------------------- #
def build_honest_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Features available at transaction time, using NO information about the label.

    Deliberately small and low-cardinality. High-cardinality categoricals
    (merchant_id, mcc) need point-in-time-correct encoding, which is Milestone 3 --
    doing it naively here would just plant a *fourth* leak. The job of this feature
    set is only to give an honest baseline to contrast against the leaky one.
    """
    ts = tx["event_ts"]
    feats = pd.DataFrame(index=tx.index)

    # Amount, log-compressed (fraud and legit spend span several orders of magnitude).
    feats["log_amount"] = np.log1p(tx["amount"].to_numpy())

    # Calendar features -- known the instant the swipe happens.
    feats["hour"] = ts.dt.hour.to_numpy()
    feats["dow"] = ts.dt.dayofweek.to_numpy()
    feats["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int).to_numpy()

    # Entry mode one-hots (CHIP / CONTACTLESS / SWIPE / ECOM / MANUAL).
    for mode in ("CHIP", "CONTACTLESS", "SWIPE", "ECOM", "MANUAL"):
        feats[f"entry_{mode}"] = (tx["entry_mode"] == mode).astype(int).to_numpy()

    # Geography and data-quality flags -- both known at swipe time.
    feats["is_foreign"] = (tx["pos_country"] != "US").astype(int).to_numpy()
    feats["mcc_missing"] = tx["mcc"].isna().astype(int).to_numpy()
    feats["is_capture"] = (tx["event_type"] == "CAPTURE").astype(int).to_numpy()

    return feats


def build_leak_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Numeric encodings of the three planted leaks, to bolt onto the honest set."""
    feats = pd.DataFrame(index=tx.index)
    # Leak A: a real reason code exists only for fraud; "NONE" otherwise.
    feats["has_dispute_code"] = (tx["dispute_reason_code"] != "NONE").astype(int).to_numpy()
    # Leaks B and C are already numeric.
    feats["merchant_fraud_rate_lifetime"] = tx["merchant_fraud_rate_lifetime"].to_numpy()
    feats["card_txn_count_next_24h"] = tx["card_txn_count_next_24h"].to_numpy()
    return feats


# --------------------------------------------------------------------------- #
# Temporal split -- the "why you can't random-split fraud data" lesson, in code.
# --------------------------------------------------------------------------- #
@dataclass
class TemporalSplit:
    """Boolean masks for a past/future split, plus the cutoff used."""

    cutoff: pd.Timestamp
    train_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_test(self) -> int:
        return int(self.test_mask.sum())


def temporal_split(tx: pd.DataFrame, train_frac: float = 0.70) -> TemporalSplit:
    """Split by TIME: earliest ``train_frac`` of the timeline trains, the rest tests.

    No row shuffling. The cutoff is the ``train_frac`` quantile of ``event_ts``,
    so the test set is strictly *later* than the training set -- exactly the
    situation a deployed model faces (predict the future from the past).
    """
    ts = tx["event_ts"]
    cutoff = ts.quantile(train_frac)
    train_mask = (ts < cutoff).to_numpy()
    test_mask = (ts >= cutoff).to_numpy()
    return TemporalSplit(cutoff=cutoff, train_mask=train_mask, test_mask=test_mask)


def _fit_auc(X: pd.DataFrame, y: np.ndarray, split: TemporalSplit) -> float:
    """Fit a balanced logistic regression on the past, score AUC on the future."""
    model = Pipeline(
        [
            # Standardise so the numeric leak features and log-amount are comparable.
            ("scale", StandardScaler()),
            # class_weight balances the 0.15%-positive problem so the model doesn't
            # trivially predict "never fraud".
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X[split.train_mask], y[split.train_mask])
    # Probability of the positive (fraud) class on the held-out future.
    proba = model.predict_proba(X[split.test_mask])[:, 1]
    return float(roc_auc_score(y[split.test_mask], proba))


# --------------------------------------------------------------------------- #
# Detectors -- one per leak archetype (plus a generic univariate screen).
# --------------------------------------------------------------------------- #
def detect_post_outcome_field(tx: pd.DataFrame, y: np.ndarray) -> dict:
    """Leak A archetype: a field that only exists *because of* the outcome.

    Signature: near-perfect class separation from a single column. Here the
    reason code is "NONE" for every legit row and a real code for every fraud,
    so its single-feature AUC is ~1.0. A field this predictive on a rare-event
    problem did not come from the transaction -- it came from the chargeback.
    """
    x = (tx["dispute_reason_code"] != "NONE").astype(int).to_numpy()
    auc = float(roc_auc_score(y, x)) if y.any() and (~y.astype(bool)).any() else float("nan")
    # Cross-tab agreement: does "code present" line up with "is fraud" exactly?
    agreement = float((x == y.astype(int)).mean())
    return {
        "column": "dispute_reason_code",
        "archetype": "post_outcome_field",
        "single_feature_auc": round(auc, 4),
        "presence_label_agreement": round(agreement, 4),
        "verdict": "LEAK" if auc > _PERFECT_AUC_FLAG else "clean",
        "why": "Reason code exists only after a chargeback is filed; it cannot be "
        "known at transaction time.",
    }


def detect_lookahead_target_encoding(tx: pd.DataFrame, y: np.ndarray) -> dict:
    """Leak B archetype: a per-group statistic computed over the WHOLE history.

    Signature: the value is *constant within each group* (here, per merchant_id)
    and equals that group's overall fraud rate. An honest as-of encoding would
    change over time as more history accrues; a value that never varies within a
    merchant proves it peeked at the full timeline (including the future).
    """
    col = "merchant_fraud_rate_lifetime"
    distinct_per_group = tx.groupby("merchant_id")[col].nunique()
    n_varying = int((distinct_per_group > 1).sum())

    # Does the stored value equal the naive full-history groupby fraud mean?
    global_rate = pd.Series(y.astype(float), index=tx.index).groupby(tx["merchant_id"]).transform("mean")
    matches_global = bool(np.allclose(tx[col].to_numpy(), global_rate.to_numpy(), atol=1e-6))

    is_leak = (n_varying == 0) and matches_global
    return {
        "column": col,
        "archetype": "lookahead_target_encoding",
        "merchants_with_varying_value": n_varying,
        "n_merchants": int(len(distinct_per_group)),
        "matches_full_history_fraud_mean": matches_global,
        "verdict": "LEAK" if is_leak else "clean",
        "why": "Constant within each merchant and equal to that merchant's lifetime "
        "fraud rate -- it averaged over transactions that had not happened yet.",
    }


def _forward_and_backward_counts(card: np.ndarray, ts: np.ndarray, horizon_h: int = 24) -> tuple:
    """Recompute, per card, how many OTHER txns fall in the next / previous window.

    Returns (forward, backward) int arrays aligned to the input order. This is the
    ground truth we compare the stored column against: if the column matches the
    FORWARD count, it looked into the future.
    """
    horizon = np.timedelta64(horizon_h, "h")
    order = np.lexsort((ts, card))  # sort by card, then time
    card_s, ts_s = card[order], ts[order]
    n = len(card)
    fwd = np.zeros(n, dtype=np.int64)
    bwd = np.zeros(n, dtype=np.int64)

    # Card block boundaries in the sorted view.
    change = np.flatnonzero(card_s[1:] != card_s[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    for s, e in zip(starts, ends, strict=True):
        block = ts_s[s:e]
        # Forward: count of later timestamps within +horizon (excludes self).
        hi = np.searchsorted(block, block + horizon, side="right")
        lo = np.searchsorted(block, block, side="right")
        fwd[s:e] = hi - lo
        # Backward: count of earlier timestamps within -horizon (excludes self).
        blo = np.searchsorted(block, block - horizon, side="left")
        bhi = np.searchsorted(block, block, side="left")
        bwd[s:e] = bhi - blo

    # Undo the sort so results line up with the original rows.
    fwd_orig = np.empty(n, dtype=np.int64)
    bwd_orig = np.empty(n, dtype=np.int64)
    fwd_orig[order] = fwd
    bwd_orig[order] = bwd
    return fwd_orig, bwd_orig


def detect_temporal_lookahead(tx: pd.DataFrame) -> dict:
    """Leak C archetype: a rolling window that counts FORWARD in time.

    Signature: the stored value matches a recomputed *forward* count (next 24h),
    not the *backward* count (previous 24h). The backward version would be a
    perfectly fine feature; the forward version uses transactions that had not
    occurred yet at scoring time.
    """
    fwd, bwd = _forward_and_backward_counts(
        tx["card_id"].to_numpy(), tx["event_ts"].to_numpy(), horizon_h=24
    )
    stored = tx["card_txn_count_next_24h"].to_numpy()
    matches_forward = float((stored == fwd).mean())
    matches_backward = float((stored == bwd).mean())
    is_leak = matches_forward > 0.999 and matches_backward < 0.999
    return {
        "column": "card_txn_count_next_24h",
        "archetype": "temporal_lookahead_window",
        "match_forward_recompute": round(matches_forward, 4),
        "match_backward_recompute": round(matches_backward, 4),
        "verdict": "LEAK" if is_leak else "clean",
        "why": "Values equal a forward-looking 24h count -- it counts transactions "
        "that occur AFTER the row being scored.",
    }


def screen_single_feature_auc(tx: pd.DataFrame, y: np.ndarray) -> dict:
    """Generic first-pass screen: univariate AUC of every candidate column.

    Teaching point: this catches Leak A loudly (~1.0) but only *whispers* about
    Leaks B and C, because look-ahead leaks need not be strong univariate
    predictors. A univariate screen is a starting point, never the whole audit.
    """
    candidates = build_honest_features(tx).join(build_leak_features(tx))
    aucs = {}
    for col in candidates.columns:
        x = candidates[col].to_numpy(dtype=float)
        if np.unique(x).size < 2:
            aucs[col] = float("nan")
            continue
        auc = roc_auc_score(y, x)
        # A protective feature (AUC < 0.5) is just as informative flipped.
        aucs[col] = round(float(max(auc, 1 - auc)), 4)
    flagged = [c for c, a in aucs.items() if not np.isnan(a) and a > _PERFECT_AUC_FLAG]
    return {"per_feature_auc": aucs, "flagged_suspiciously_perfect": flagged}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class AuditResult:
    n_rows: int
    row_fraud_rate: float
    split_cutoff: str
    n_train: int
    n_test: int
    auc_with_leaks: float
    auc_without_leaks: float
    auc_gap: float
    detectors: list = field(default_factory=list)
    univariate_screen: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def run_audit(tx: pd.DataFrame, labels: pd.DataFrame) -> AuditResult:
    """Full Milestone 2 audit: detect the three leaks and quantify the AUC gap."""
    # Row-level label (eventual truth) via auth_code membership -- see _labeling.
    y = _labeling.attach_row_label(tx, labels).to_numpy()

    # Detect each planted leak with its archetype-specific test.
    detectors = [
        detect_post_outcome_field(tx, y),
        detect_lookahead_target_encoding(tx, y),
        detect_temporal_lookahead(tx),
    ]
    screen = screen_single_feature_auc(tx, y)

    # Quantify the damage on a strict past->future split.
    split = temporal_split(tx)
    honest = build_honest_features(tx)
    leaky = honest.join(build_leak_features(tx))
    auc_honest = _fit_auc(honest, y, split)
    auc_leaky = _fit_auc(leaky, y, split)

    return AuditResult(
        n_rows=int(len(tx)),
        row_fraud_rate=round(float(y.mean()), 6),
        split_cutoff=str(split.cutoff),
        n_train=split.n_train,
        n_test=split.n_test,
        auc_with_leaks=round(auc_leaky, 4),
        auc_without_leaks=round(auc_honest, 4),
        auc_gap=round(auc_leaky - auc_honest, 4),
        detectors=detectors,
        univariate_screen=screen,
    )


# --------------------------------------------------------------------------- #
# Report writer + CLI
# --------------------------------------------------------------------------- #
def _render_markdown(r: AuditResult) -> str:
    lines = [
        "# Leakage Audit Report (Milestone 2)",
        "",
        "> Auto-generated by `python -m lonestar.audit`. Do not edit by hand.",
        "",
        "## Headline",
        "",
        f"- Rows audited: **{r.n_rows:,}**  |  row-level fraud rate: **{r.row_fraud_rate:.4%}**",
        f"- Temporal split at **{r.split_cutoff}** "
        f"(train {r.n_train:,} rows in the past, test {r.n_test:,} rows in the future)",
        f"- Test AUC **with** the three leaks: **{r.auc_with_leaks:.4f}**  (a fantasy)",
        f"- Test AUC **without** them: **{r.auc_without_leaks:.4f}**  (honest)",
        f"- **Inflation from leakage: {r.auc_gap:+.4f} AUC**",
        "",
        "The leaks do not make the model better -- they make it *lie*. Removing them "
        "is the point, and the honest number is the one you can defend in production.",
        "",
        "## Per-leak findings",
        "",
    ]
    for d in r.detectors:
        lines.append(f"### `{d['column']}` — {d['archetype']}  → **{d['verdict']}**")
        lines.append("")
        for k, v in d.items():
            if k in ("column", "archetype", "verdict", "why"):
                continue
            lines.append(f"- {k}: `{v}`")
        lines.append(f"- _why it leaks_: {d['why']}")
        lines.append("")
    lines.append("## Univariate screen (why one test is not enough)")
    lines.append("")
    lines.append(
        "Single-feature AUC flags the post-outcome field loudly but barely reacts to "
        "the look-ahead leaks -- proof that structural detectors, not just univariate "
        "screening, are required."
    )
    lines.append("")
    flagged = r.univariate_screen.get("flagged_suspiciously_perfect", [])
    lines.append(f"- Columns with suspiciously perfect AUC (>{_PERFECT_AUC_FLAG}): `{flagged}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Milestone 2 leakage audit.")
    ap.add_argument("--data", default="data/raw", help="Directory with the parquet files.")
    ap.add_argument("--out", default="docs", help="Where to write the markdown report.")
    ap.add_argument(
        "--json-out", default="reports/leakage_audit.json", help="Machine-readable result."
    )
    args = ap.parse_args()

    data = Path(args.data)
    tx = pd.read_parquet(data / "transactions.parquet")
    labels = pd.read_parquet(data / "chargeback_labels.parquet")

    result = run_audit(tx, labels)

    # Write the human-readable report.
    out_md = Path(args.out) / "leakage_audit_report.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(result), encoding="utf-8")

    # Write the machine-readable result (for CI assertions / dashboards).
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")

    # Console summary.
    print(json.dumps(result.as_dict(), indent=2))
    print(f"\nWrote {out_md} and {json_out}")


if __name__ == "__main__":
    main()
