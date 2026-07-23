"""Milestone 9 -- challenger/champion retraining.

This is the automated response to the drift alert from Milestone 8. When the
monitor says the world has changed, something has to actually retrain the model --
carefully, with guardrails, and with a written record of why a new model was or
was not promoted.

Two constraints shape the whole design:

1. **Label maturity.** Chargebacks arrive 30-60 days late. You cannot train on
   last week's transactions, because most of their fraud has not been reported yet
   -- those rows would be silently labelled "not fraud" and teach the model
   exactly the wrong thing. So training data is cut at ``as_of - LABEL_MATURITY_DAYS``.
   A ring that starts in month 13 cannot be learned until roughly month 15. The
   monitor buys you early warning; it does not remove the wait.

2. **Never promote on a whim.** A challenger replaces the champion only if it
   wins on the honest metric by a margin, on a held-out window with enough fraud
   to be meaningful, without regressing any individual channel. Otherwise the
   champion stays and the run is recorded as a HOLD.

The timeline of a cycle, all in transaction time:

    [-------------- challenger trains --------------|-- eval --|-- immature --]
                                                              ^            ^
                                          maturity cutoff ----+     as_of --+
      (as_of - LABEL_MATURITY_DAYS - EVAL_WINDOW_DAYS)

Run it:  ``python -m lonestar.retraining --data data/raw --models models``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from lonestar import features, modeling
from lonestar.decisions import _SEGMENT_COL

# Chargebacks land 30-60 days after the transaction, so anything newer than this
# has labels we cannot trust yet. Matches the generator's label_delay_max.
LABEL_MATURITY_DAYS = 60

# The most recent matured slice is held out to compare the two models on.
EVAL_WINDOW_DAYS = 60

# --- Promotion guardrails ---------------------------------------------------
# The challenger must beat the champion by a real margin, not noise.
MIN_RELATIVE_IMPROVEMENT = 0.05  # +5% PR-AUC, relative
# ...and must not quietly wreck a channel to win on average.
MAX_SEGMENT_REGRESSION = 0.05  # absolute PR-AUC drop allowed in any segment
# ...and the comparison must rest on enough fraud to mean anything.
MIN_EVAL_POSITIVES = 20


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def maturity_cutoff(as_of: pd.Timestamp, days: int = LABEL_MATURITY_DAYS) -> pd.Timestamp:
    """Transactions after this instant do not yet have trustworthy labels."""
    return as_of - np.timedelta64(days, "D")


def training_windows(
    as_of: pd.Timestamp,
    maturity_days: int = LABEL_MATURITY_DAYS,
    eval_days: int = EVAL_WINDOW_DAYS,
) -> dict:
    """Split transaction time into (train, eval, immature) for one cycle."""
    mature_end = maturity_cutoff(as_of, maturity_days)
    train_end = mature_end - np.timedelta64(eval_days, "D")
    return {"train_end": train_end, "eval_start": train_end, "eval_end": mature_end}


# --------------------------------------------------------------------------- #
# Training a candidate as-of a point in time
# --------------------------------------------------------------------------- #
def train_as_of(
    frame: pd.DataFrame, train_end: pd.Timestamp, kind: str = "xgb", label_col: str = "is_fraud"
):
    """Fit a model on everything strictly before ``train_end``.

    ``frame`` is the feature table (already label-censored by the caller, so the
    labels here are only those that had actually been reported).
    """
    train = frame[frame["event_ts"] < train_end]
    X = train[list(features.FEATURE_COLUMNS)]
    y = train[label_col].to_numpy()
    if y.sum() == 0:
        raise ValueError("no positive labels in the training window")
    model = modeling.build_model(kind, modeling._pos_weight(y))
    model.fit(X, y)
    return model, {"n_rows": int(len(train)), "n_fraud": int(y.sum()), "train_end": str(train_end)}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    if y.sum() == 0 or y.sum() == len(y):
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    return {
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def evaluate_model(model, eval_df: pd.DataFrame, label_col: str = "is_fraud") -> dict:
    """Overall and per-segment metrics on the held-out evaluation window."""
    X = eval_df[list(features.FEATURE_COLUMNS)]
    y = eval_df[label_col].to_numpy()
    proba = model.predict_proba(X)[:, 1]

    per_segment = {}
    if _SEGMENT_COL in eval_df.columns:
        for seg, g in eval_df.groupby(_SEGMENT_COL):
            yy = g[label_col].to_numpy()
            if yy.sum() < 5:  # too few positives to score this channel
                continue
            pp = model.predict_proba(g[list(features.FEATURE_COLUMNS)])[:, 1]
            per_segment[str(seg)] = _metrics(yy, pp)

    return {
        "overall": _metrics(y, proba),
        "per_segment": per_segment,
        "n_rows": int(len(eval_df)),
        "n_positives": int(y.sum()),
    }


# --------------------------------------------------------------------------- #
# The promotion decision
# --------------------------------------------------------------------------- #
@dataclass
class PromotionDecision:
    promote: bool
    reasons: list = field(default_factory=list)
    champion_pr_auc: float = float("nan")
    challenger_pr_auc: float = float("nan")
    relative_improvement: float = float("nan")
    segment_regressions: dict = field(default_factory=dict)


def promotion_decision(champion_eval: dict, challenger_eval: dict) -> PromotionDecision:
    """Apply every guardrail. Promote only if all of them pass.

    The guardrails exist because "the new model scored higher" is not sufficient
    evidence to put it in front of live money: the win must be material, it must
    rest on enough fraud cases, and it must not come from sacrificing a channel.
    """
    reasons: list[str] = []
    champ = champion_eval["overall"]["pr_auc"]
    chall = challenger_eval["overall"]["pr_auc"]
    n_pos = challenger_eval["n_positives"]

    rel = (chall - champ) / champ if champ and champ > 0 else float("nan")

    # Guardrail 1: enough fraud in the evaluation window to decide on.
    enough_data = n_pos >= MIN_EVAL_POSITIVES
    if not enough_data:
        reasons.append(
            f"only {n_pos} fraud cases in the eval window (need {MIN_EVAL_POSITIVES})"
        )

    # Guardrail 2: a material improvement, not noise.
    material = np.isfinite(rel) and rel >= MIN_RELATIVE_IMPROVEMENT
    if not material:
        reasons.append(
            f"PR-AUC {champ:.4f} -> {chall:.4f} ({rel:+.1%}); "
            f"need >= {MIN_RELATIVE_IMPROVEMENT:+.0%}"
        )
    else:
        reasons.append(f"PR-AUC {champ:.4f} -> {chall:.4f} ({rel:+.1%})")

    # Guardrail 3: no channel quietly sacrificed to win on average.
    regressions = {}
    for seg, m in champion_eval["per_segment"].items():
        new = challenger_eval["per_segment"].get(seg)
        if not new:
            continue
        drop = m["pr_auc"] - new["pr_auc"]
        if drop > MAX_SEGMENT_REGRESSION:
            regressions[seg] = round(drop, 4)
    if regressions:
        reasons.append(f"segment regressions beyond {MAX_SEGMENT_REGRESSION}: {regressions}")

    promote = bool(enough_data and material and not regressions)
    return PromotionDecision(
        promote=promote,
        reasons=reasons,
        champion_pr_auc=champ,
        challenger_pr_auc=chall,
        relative_improvement=round(rel, 4) if np.isfinite(rel) else None,
        segment_regressions=regressions,
    )


# --------------------------------------------------------------------------- #
# One full cycle
# --------------------------------------------------------------------------- #
def run_cycle(
    tx: pd.DataFrame,
    labels: pd.DataFrame,
    champion,
    as_of: pd.Timestamp | None = None,
    kind: str = "xgb",
) -> dict:
    """Train a challenger, compare it to the champion, decide, and report.

    Labels are censored at ``as_of`` -- only chargebacks actually reported by then
    are visible, exactly as they would be on the day the job runs.
    """
    as_of = as_of or tx["event_ts"].max()
    windows = training_windows(as_of)

    # Build features with as-of censored labels: the honest view on that date.
    frame = features.build_training_frame(tx, labels, as_of=as_of)
    frame = frame.merge(
        tx[["transaction_id", _SEGMENT_COL]], on="transaction_id", how="left"
    )

    challenger, train_info = train_as_of(frame, windows["train_end"], kind=kind)

    eval_df = frame[
        (frame["event_ts"] >= windows["eval_start"]) & (frame["event_ts"] < windows["eval_end"])
    ]
    champion_eval = evaluate_model(champion, eval_df)
    challenger_eval = evaluate_model(challenger, eval_df)
    decision = promotion_decision(champion_eval, challenger_eval)

    return {
        "as_of": str(as_of),
        "windows": {k: str(v) for k, v in windows.items()},
        "label_maturity_days": LABEL_MATURITY_DAYS,
        "challenger_training": train_info,
        "champion_eval": champion_eval,
        "challenger_eval": challenger_eval,
        "decision": asdict(decision),
        "challenger": challenger,  # in-memory; the CLI persists it if promoted
        "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _render_markdown(rep: dict) -> str:
    d = rep["decision"]
    verdict = "**PROMOTE**" if d["promote"] else "**HOLD** (champion stays)"
    seg_rows = []
    for seg, m in sorted(rep["champion_eval"]["per_segment"].items()):
        new = rep["challenger_eval"]["per_segment"].get(seg, {})
        seg_rows.append(
            f"| {seg} | {m['pr_auc']:.4f} | {new.get('pr_auc', float('nan')):.4f} |"
        )
    return "\n".join(
        [
            "# Retraining Cycle Report (Milestone 9)",
            "",
            "> Auto-generated by `python -m lonestar.retraining`. Do not edit by hand.",
            "",
            f"## Verdict: {verdict}",
            "",
            *[f"- {r}" for r in d["reasons"]],
            "",
            "## Why the training window stops where it does",
            "",
            f"Chargebacks arrive 30-60 days late, so anything after "
            f"`{rep['windows']['eval_end']}` has labels we cannot trust yet — those rows "
            "would look fraud-free simply because nobody has reported them. Training "
            f"therefore stops at `{rep['windows']['train_end']}`, and the most recent "
            "**matured** slice is held out to compare the two models on.",
            "",
            "This is the honest cost of a delayed label: the drift monitor (M8) can warn "
            "you the day an attack starts, but a *retrained model* still has to wait for "
            "the chargebacks. The gap is covered operationally, not statistically — see "
            "the runbook.",
            "",
            "| Window | Boundary |",
            "|---|---|",
            f"| Challenger trains before | `{rep['windows']['train_end']}` |",
            f"| Evaluation window | `{rep['windows']['eval_start']}` → `{rep['windows']['eval_end']}` |",
            f"| Labels immature after | `{rep['windows']['eval_end']}` |",
            "",
            "## Head-to-head (held-out, matured labels)",
            "",
            f"Evaluation window: **{rep['challenger_eval']['n_rows']:,} transactions**, "
            f"**{rep['challenger_eval']['n_positives']} fraud cases**.",
            "",
            "| Model | PR-AUC | ROC-AUC |",
            "|---|---|---|",
            f"| Champion | {rep['champion_eval']['overall']['pr_auc']:.4f} | "
            f"{rep['champion_eval']['overall']['roc_auc']:.4f} |",
            f"| Challenger | {rep['challenger_eval']['overall']['pr_auc']:.4f} | "
            f"{rep['challenger_eval']['overall']['roc_auc']:.4f} |",
            "",
            "### Per channel (the regression guardrail)",
            "",
            "| Segment | Champion PR-AUC | Challenger PR-AUC |",
            "|---|---|---|",
            *seg_rows,
            "",
            "## The guardrails",
            "",
            "A challenger is promoted only if **all** hold:",
            "",
            f"1. At least **{MIN_EVAL_POSITIVES} fraud cases** in the evaluation window — "
            "a PR-AUC computed on a handful of positives is noise.",
            f"2. At least **{MIN_RELATIVE_IMPROVEMENT:+.0%} relative PR-AUC** improvement — "
            "a new model must earn its deployment, not tie.",
            f"3. **No channel** loses more than **{MAX_SEGMENT_REGRESSION}** absolute PR-AUC — "
            "an average can improve while a whole segment is quietly sacrificed.",
            "",
            "If any guardrail fails the run is recorded as a HOLD and the champion stays "
            "in production. A retraining pipeline that always promotes is not a pipeline, "
            "it is a liability.",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one challenger/champion cycle (M9).")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--models", default="models")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--as-of", default=None, help="ISO timestamp; defaults to latest transaction.")
    ap.add_argument(
        "--champion-as-of",
        default=None,
        help=(
            "Simulate a STALE champion trained as-of this date instead of loading the "
            "saved one. Useful for demonstrating the cycle: a champion trained before "
            "a regime change should lose to a challenger trained after it."
        ),
    )
    ap.add_argument(
        "--promote",
        action="store_true",
        help="Actually overwrite the champion artifact when the decision is PROMOTE.",
    )
    args = ap.parse_args()

    data, models_dir = Path(args.data), Path(args.models)
    tx = pd.read_parquet(data / "transactions.parquet")
    labels = pd.read_parquet(data / "chargeback_labels.parquet")
    if args.champion_as_of:
        champ_as_of = pd.Timestamp(args.champion_as_of)
        champ_frame = features.build_training_frame(tx, labels, as_of=champ_as_of)
        champion, champ_info = train_as_of(
            champ_frame, training_windows(champ_as_of)["train_end"]
        )
        print(f"simulated stale champion: {champ_info}")
    else:
        champion = joblib.load(models_dir / "fraud_model.joblib")

    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    rep = run_cycle(tx, labels, champion, as_of=as_of)
    challenger = rep.pop("challenger")

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "retraining_cycle.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    doc = Path(args.docs) / "retraining.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(_render_markdown(rep), encoding="utf-8")

    d = rep["decision"]
    print("PROMOTE" if d["promote"] else "HOLD")
    for r in d["reasons"]:
        print(f"  - {r}")

    if d["promote"] and args.promote:
        joblib.dump(challenger, models_dir / "fraud_model.joblib")
        print(f"champion replaced at {models_dir / 'fraud_model.joblib'}")
    elif d["promote"]:
        print("(re-run with --promote to actually replace the champion)")

    print(f"\nWrote {doc} and {reports / 'retraining_cycle.json'}")


if __name__ == "__main__":
    main()
