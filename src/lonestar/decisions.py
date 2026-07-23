"""Milestone 5 -- the decision layer (turning probabilities into dollars).

A model outputs a probability. A business needs a *decision*: approve or decline.
The bridge is a threshold -- and choosing it at the default 0.5 is almost always
wrong for fraud, because the two mistakes cost wildly different amounts:

    * Approve a fraud (false negative)  -> you eat the whole transaction amount.
    * Decline a good customer (false positive) -> you lose a little margin plus
      some goodwill, but nothing like the full amount.

Because the two mistakes cost such different amounts, the right threshold is the
one that minimises expected DOLLARS, not the one that minimises the error count
(which is all the default 0.5 does). Where that dollar-optimal threshold lands
depends on how the model is calibrated: our Milestone-4 model is imbalance-weighted
(``scale_pos_weight``), so it already scores fraud aggressively -- and the
cost-optimal point comes out slightly ABOVE 0.5, trading a sliver of fraud capture
for a large cut in costly false declines. This module finds that point by
minimising dollars, does it PER SEGMENT (entry mode), and writes an economics memo
plus a servable policy file.

Two disciplines carry over from earlier milestones:
  * Thresholds are tuned on the PAST (train window) and the savings are reported
    on the FUTURE (test window) -- you never tune a threshold on the data you
    grade it against.
  * The label is the honest row-level fraud flag from the feature table.

Run it:  ``python -m lonestar.decisions --features data/features/features.parquet \
             --raw data/raw --model models/fraud_model.joblib --out models``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lonestar.features import FEATURE_COLUMNS
from lonestar.modeling import TRAIN_FRAC as _MODEL_TRAIN_FRAC

_SEGMENT_COL = "entry_mode"

# --- The three-way temporal split (see ADR 0004) ------------------------------
# Thresholds MUST NOT be tuned on data the model was trained on. An overfit model
# scores its own training fraud near 1.0, so an in-sample threshold looks great
# and does not transfer -- at full data scale it collapses to "decline nobody".
# So the timeline is cut three ways:
#     [0, TRAIN_FRAC)            the model's training window   (M4)
#     [TRAIN_FRAC, _CALIB_END)   held-out CALIBRATION -> tune thresholds here
#     [_CALIB_END, 1.0)          held-out TEST -> report economics here
_CALIB_END_FRAC = 0.90


# --------------------------------------------------------------------------- #
# The cost model -- the asymmetry that makes 0.5 the wrong threshold
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """Dollar cost of each mistake.

    * A missed fraud costs the full transaction amount (you refund the cardholder
      and eat the chargeback).
    * A false decline costs a fixed friction charge (support + goodwill) plus a
      small fraction of the amount (the margin you would have earned on the sale).
    Correct decisions cost nothing.
    """

    fp_fixed: float = 5.0  # $ friction per wrongly declined customer
    fp_rate: float = 0.02  # + fraction of amount (lost margin) per false decline

    def false_negative_cost(self, amount: np.ndarray) -> np.ndarray:
        return amount  # approving fraud loses the whole amount

    def false_positive_cost(self, amount: np.ndarray) -> np.ndarray:
        return self.fp_fixed + self.fp_rate * amount


def expected_cost(
    y: np.ndarray, proba: np.ndarray, amount: np.ndarray, threshold: float, cost: CostModel
) -> float:
    """Total dollar cost of declining every txn with score >= threshold."""
    decline = proba >= threshold
    # False negatives: approved (score below threshold) but actually fraud.
    fn = (~decline) & (y == 1)
    # False positives: declined but actually legitimate.
    fp = decline & (y == 0)
    return float(cost.false_negative_cost(amount[fn]).sum() + cost.false_positive_cost(amount[fp]).sum())


def _threshold_grid(proba: np.ndarray) -> np.ndarray:
    """Candidate thresholds adapted to the score distribution, plus 0.5 and ends."""
    qs = np.quantile(proba, np.linspace(0.0, 1.0, 201))
    grid = np.unique(np.concatenate([qs, [0.0, 0.5, 1.0 + 1e-9]]))
    return grid


def optimal_threshold(
    y: np.ndarray, proba: np.ndarray, amount: np.ndarray, cost: CostModel
) -> tuple[float, float]:
    """Return (threshold, cost) minimising expected dollars over the grid."""
    grid = _threshold_grid(proba)
    costs = np.array([expected_cost(y, proba, amount, t, cost) for t in grid])
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


# --------------------------------------------------------------------------- #
# Policy: a threshold per segment (entry mode)
# --------------------------------------------------------------------------- #
def fit_segment_policy(
    df: pd.DataFrame, cost: CostModel, segment_col: str = _SEGMENT_COL
) -> dict:
    """Tune one cost-optimal threshold per segment, plus a global fallback."""
    y = df["is_fraud"].to_numpy()
    proba = df["proba"].to_numpy()
    amount = df["amount"].to_numpy()

    global_t, _ = optimal_threshold(y, proba, amount, cost)
    seg_thresholds: dict[str, float] = {}
    for seg, g in df.groupby(segment_col):
        # A segment needs both classes present to tune; else use the global one.
        yy = g["is_fraud"].to_numpy()
        if yy.sum() == 0 or yy.sum() == len(yy):
            seg_thresholds[str(seg)] = global_t
            continue
        t, _ = optimal_threshold(yy, g["proba"].to_numpy(), g["amount"].to_numpy(), cost)
        seg_thresholds[str(seg)] = t

    return {
        "segment_col": segment_col,
        "global_threshold": global_t,
        "segment_thresholds": seg_thresholds,
        "cost_model": {"fp_fixed": cost.fp_fixed, "fp_rate": cost.fp_rate},
    }


def apply_policy(df: pd.DataFrame, policy: dict) -> np.ndarray:
    """Return a boolean 'decline' decision per row using the per-segment policy."""
    seg = df[policy["segment_col"]].astype(str).to_numpy()
    proba = df["proba"].to_numpy()
    thresholds = np.array(
        [policy["segment_thresholds"].get(s, policy["global_threshold"]) for s in seg]
    )
    return proba >= thresholds


# --------------------------------------------------------------------------- #
# Economics evaluation on the held-out future
# --------------------------------------------------------------------------- #
def _operating_stats(y, decline, amount, cost) -> dict:
    fn = (~decline) & (y == 1)
    fp = decline & (y == 0)
    tp = decline & (y == 1)
    fraud_total = float(amount[y == 1].sum())
    return {
        "total_cost": round(expected_cost_from_masks(fn, fp, amount, cost), 2),
        "fraud_dollars_caught": round(float(amount[tp].sum()), 2),
        "fraud_dollars_missed": round(float(amount[fn].sum()), 2),
        "fraud_dollars_total": round(fraud_total, 2),
        "fraud_recall_dollars": round(float(amount[tp].sum()) / fraud_total, 4) if fraud_total else 0.0,
        "false_declines": int(fp.sum()),
        "false_decline_rate": round(float(fp.sum()) / int((y == 0).sum()), 5),
    }


def expected_cost_from_masks(fn, fp, amount, cost) -> float:
    return float(cost.false_negative_cost(amount[fn]).sum() + cost.false_positive_cost(amount[fp]).sum())


def evaluate(df_test: pd.DataFrame, policy: dict, cost: CostModel) -> dict:
    """Compare four policies on the test window and quantify the savings."""
    y = df_test["is_fraud"].to_numpy()
    proba = df_test["proba"].to_numpy()
    amount = df_test["amount"].to_numpy()

    # Baseline 1: approve everything (the do-nothing cost = all fraud lost).
    approve_all = np.zeros(len(df_test), dtype=bool)
    # Baseline 2: the naive default threshold.
    default_05 = proba >= 0.5
    # Global optimal threshold.
    global_dec = proba >= policy["global_threshold"]
    # Per-segment policy.
    segment_dec = apply_policy(df_test, policy)

    results = {
        "approve_all": _operating_stats(y, approve_all, amount, cost),
        "threshold_0.5": _operating_stats(y, default_05, amount, cost),
        "global_optimal": _operating_stats(y, global_dec, amount, cost),
        "per_segment": _operating_stats(y, segment_dec, amount, cost),
    }
    base = results["approve_all"]["total_cost"]
    for k in results:
        results[k]["savings_vs_approve_all"] = round(base - results[k]["total_cost"], 2)
    return results


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class DecisionReport:
    policy: dict = field(default_factory=dict)
    economics: dict = field(default_factory=dict)
    split_cutoff: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _score_frame(features_path: Path, raw_dir: Path, model_path: Path) -> pd.DataFrame:
    """Join features + raw amount/segment, score with the model."""
    feat = pd.read_parquet(features_path)
    tx = pd.read_parquet(raw_dir / "transactions.parquet")[["transaction_id", "amount", _SEGMENT_COL]]
    df = feat.merge(tx, on="transaction_id", how="left")
    model = joblib.load(model_path)
    df["proba"] = model.predict_proba(df[list(FEATURE_COLUMNS)])[:, 1]
    return df


def three_way_split(ts: pd.Series) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (calibration_mask, test_mask, cutoffs) for honest threshold tuning.

    The calibration window starts where the MODEL's training window ends, so no
    threshold is ever tuned on a row the model memorised.
    """
    train_end = ts.quantile(_MODEL_TRAIN_FRAC)
    calib_end = ts.quantile(_CALIB_END_FRAC)
    calib = ((ts >= train_end) & (ts < calib_end)).to_numpy()
    test = (ts >= calib_end).to_numpy()
    return calib, test, {"model_train_end": str(train_end), "calib_end": str(calib_end)}


def run(features_path: Path, raw_dir: Path, model_path: Path, cost: CostModel | None = None) -> DecisionReport:
    cost = cost or CostModel()
    df = _score_frame(features_path, raw_dir, model_path)

    # Tune on a HELD-OUT calibration window; report economics on a later test window.
    calib_mask, test_mask, cutoffs = three_way_split(df["event_ts"])
    calib = df[calib_mask]
    test = df[test_mask]

    policy = fit_segment_policy(calib, cost)
    economics = evaluate(test, policy, cost)
    return DecisionReport(policy=policy, economics=economics, split_cutoff=json.dumps(cutoffs))


def _render_memo(r: DecisionReport) -> str:
    e = r.economics
    p = r.policy
    seg_lines = "\n".join(
        f"| {seg} | {t:.4f} |" for seg, t in sorted(p["segment_thresholds"].items())
    )

    def stat(name, key):
        s = e[key]
        return (
            f"| {name} | ${s['total_cost']:,.0f} | ${s['savings_vs_approve_all']:,.0f} | "
            f"{s['fraud_recall_dollars']:.1%} | {s['false_decline_rate']:.3%} |"
        )

    return "\n".join(
        [
            "# Economics Memo (Milestone 5)",
            "",
            "> Auto-generated by `python -m lonestar.decisions`. Do not edit by hand.",
            "",
            "## The core idea",
            "",
            "A missed fraud costs the **whole transaction amount**; a false decline "
            f"costs only **${p['cost_model']['fp_fixed']:.0f} + "
            f"{p['cost_model']['fp_rate']:.0%} of the amount**. The default 0.5 "
            "threshold minimises the *error count*; what a business actually wants is "
            "the threshold that minimises *dollars*. This memo finds that point.",
            "",
            f"Here the cost-optimal global threshold is **{p['global_threshold']:.4f}**. "
            "Thresholds are tuned on a **held-out calibration window** — the slice of "
            "time *after* the model's training window — and the economics below are "
            "measured on a later **test window** the thresholds never saw. That three-way "
            "split matters enormously: tuning thresholds on the model's own training data "
            "reads its memorised, near-perfect in-sample scores and produces a threshold "
            "that does not transfer (see ADR 0004).",
            "",
            f"Split cutoffs: `{r.split_cutoff}`",
            "",
            "## Policy comparison (test window)",
            "",
            "| Policy | Total cost | Saved vs approve-all | Fraud $ caught | False-decline rate |",
            "|---|---|---|---|---|",
            stat("Approve everything", "approve_all"),
            stat("Default threshold 0.5", "threshold_0.5"),
            stat("Global optimal threshold", "global_optimal"),
            stat("Per-segment thresholds", "per_segment"),
            "",
            "## Per-segment thresholds",
            "",
            "Each entry mode gets its own threshold, and the result is more interesting "
            "than a uniform tightening. Channels where fraud is **predictable** (the "
            "e-commerce channel the fraud ring attacks) get an aggressive, low threshold. "
            "Channels whose fraud is unpredictable background noise get a threshold at or "
            "near **1.0 — meaning 'never decline'**, because declining there burns "
            "goodwill and money without catching anything the model can actually see.",
            "",
            "That is the cost model doing real work: it does not just ask *how likely is "
            "fraud*, it asks *where is intervening worth the money*.",
            "",
            f"| {p['segment_col']} | threshold |",
            "|---|---|",
            seg_lines,
            "",
            "## Why this matters",
            "",
            "This is the step that turns a PR-AUC into a number a business cares "
            "about. The model is unchanged from Milestone 4; the entire gain here is "
            "decision policy — choosing where to cut the score, in dollars, per "
            "channel. A natural extension is a three-way approve / **review** / decline "
            "policy that routes the uncertain middle band to human analysts.",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the cost-optimal decision policy (M5).")
    ap.add_argument("--features", default="data/features/features.parquet")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--model", default="models/fraud_model.joblib")
    ap.add_argument("--out", default="models")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--fp-fixed", type=float, default=5.0)
    ap.add_argument("--fp-rate", type=float, default=0.02)
    args = ap.parse_args()

    cost = CostModel(fp_fixed=args.fp_fixed, fp_rate=args.fp_rate)
    report = run(Path(args.features), Path(args.raw), Path(args.model), cost)

    # Servable policy artifact (M6 will load this alongside the model).
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision_policy.json").write_text(json.dumps(report.policy, indent=2), encoding="utf-8")

    # Economics memo.
    doc = Path(args.docs) / "economics_memo.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(_render_memo(report), encoding="utf-8")

    print(json.dumps(report.as_dict(), indent=2))
    print(f"\nWrote {out / 'decision_policy.json'} and {doc}")


if __name__ == "__main__":
    main()
