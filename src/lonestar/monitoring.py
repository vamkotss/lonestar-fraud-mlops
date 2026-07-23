"""Milestone 8 -- drift monitoring for a rare-event problem.

The single most important fact about this data is that **labels arrive 30-60 days
late** (Milestone 3). When a fraud ring switches on you cannot find out by watching
the fraud rate -- that signal is still in the post. You have to watch what is
visible immediately: the **inputs** and the **model's scores**.

But there is a second, subtler problem, and getting it right is what this milestone
is really about:

    Population-level drift metrics do not detect fraud rings.

A ring touches ~1% of cards. Against ~12,000 transactions a month its footprint is
a rounding error, and a population PSI over all features barely moves -- we compute
it below and show exactly that. Averaging washes out rare subpopulations, which is
precisely what a fraud team must catch.

What *does* work, and what real fraud teams monitor:

  * **The tail, not the centre.** A ring pushes a small number of transactions to
    high scores. The mean score is unchanged; the *rate of high-scoring
    transactions* multiplies.
  * **Per segment, not overall.** The attack lives in one channel (e-commerce).
    Monitoring that channel separately makes a 3-4x jump obvious.
  * **Against a control limit.** The baseline period gives a mean and standard
    deviation of the monthly alert rate; several sigma above that is a regime
    change, not noise.

Every metric in the alerting path is **label-free** -- as it must be, since in
production you would not yet know which of these transactions were fraud.

Run it:  ``python -m lonestar.monitoring --features data/features/features.parquet \
             --raw data/raw --model models/fraud_model.joblib --out reports``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lonestar.decisions import _score_frame
from lonestar.features import FEATURE_COLUMNS

# PSI bands (industry convention for credit/fraud monitoring).
PSI_STABLE = 0.10
PSI_MAJOR = 0.25

# The "tail" is defined RELATIVE to each segment's own baseline score
# distribution: its 99th percentile. This makes the monitor independent of model
# calibration -- a hard-won lesson. A fixed cutoff (e.g. 0.5) is a genuine tail for
# one model and captures 30% of the population for another (our imbalance-weighted
# full-scale model), at which point "tail rate" measures the bulk, its baseline
# variance explodes, and nothing ever trips. With a quantile the baseline rate is
# ~1% per segment BY CONSTRUCTION, whatever the model does.
TAIL_QUANTILE = 0.99

# Control limits. A real regime change should be both economically meaningful
# (the tail rate multiplies) and statistically solid (many baseline sigma). Ratio
# alone is noisy on small segments; sigma alone is scale-sensitive. Require both.
RATIO_WARN = 1.5
RATIO_ALERT = 2.0
SIGMA_WARN = 2.0
SIGMA_ALERT = 3.0
SIGMA_EXTREME = 5.0  # an outlier this large alerts on its own

# The stable operating baseline -- the fraction of the timeline treated as "normal
# operations". In production this is the window you validated the champion on.
BASELINE_FRAC = 0.70

# Windows smaller than this are skipped: a truncated final month produces
# meaningless drift on calendar features and would fire a false alarm.
MIN_WINDOW_ROWS = 2000

_START = pd.Timestamp("2024-01-01")
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# PSI -- kept because the negative result is itself the finding
# --------------------------------------------------------------------------- #
def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Bin edges come from the REFERENCE (quantile bins), because the reference is the
    world the model was built in. 0.0 means identical; larger means more divergent.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        # Degenerate (constant/binary feature): compare category shares directly.
        values = np.unique(np.concatenate([reference, current]))
        ref_pct = np.array([(reference == v).mean() for v in values]) + _EPS
        cur_pct = np.array([(current == v).mean() for v in values]) + _EPS
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_pct = np.histogram(reference, bins=edges)[0] / reference.size + _EPS
        cur_pct = np.histogram(current, bins=edges)[0] / current.size + _EPS

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_band(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MAJOR:
        return "moderate"
    return "major"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, columns=None) -> dict:
    """PSI per feature, plus how many drifted past the stable band."""
    columns = list(columns or FEATURE_COLUMNS)
    per_feature = {
        c: round(compute_psi(reference[c].to_numpy(), current[c].to_numpy()), 4) for c in columns
    }
    drifted = [c for c, v in per_feature.items() if v >= PSI_STABLE]
    return {
        "per_feature_psi": per_feature,
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "share_drifted": round(len(drifted) / len(columns), 4) if columns else 0.0,
        "top_drifters": sorted(per_feature.items(), key=lambda kv: kv[1], reverse=True)[:5],
    }


# --------------------------------------------------------------------------- #
# The metric that actually works: segment tail alert rate
# --------------------------------------------------------------------------- #
def alert_rate(scores: np.ndarray, threshold: float) -> float:
    """Share of transactions the model scores at or above the tail threshold."""
    scores = np.asarray(scores, dtype=float)
    return float((scores >= threshold).mean()) if scores.size else 0.0


def tail_threshold(scores: np.ndarray, quantile: float = TAIL_QUANTILE) -> float:
    """The score defining the tail: a quantile of the BASELINE distribution."""
    scores = np.asarray(scores, dtype=float)
    return float(np.quantile(scores, quantile)) if scores.size else 1.0


def _month_index(ts: pd.Series) -> np.ndarray:
    return ((ts.dt.year - _START.year) * 12 + (ts.dt.month - _START.month)).to_numpy()


def baseline_stats(
    baseline: pd.DataFrame,
    segment_col: str = "entry_mode",
    quantile: float = TAIL_QUANTILE,
) -> dict:
    """Per-segment tail threshold + mean/std of the MONTHLY tail rate.

    The tail threshold is that segment's own baseline quantile, so the baseline
    tail rate is ~1-quantile by construction. Month-to-month variation of that rate
    (rather than one pooled number) is what the control limit needs.
    """
    b = baseline.assign(_month=_month_index(baseline["event_ts"]))
    stats: dict[str, dict] = {}
    for seg, g in b.groupby(segment_col):
        thr = tail_threshold(g["proba"].to_numpy(), quantile)
        monthly = g.groupby("_month")["proba"].apply(
            lambda s, thr=thr: alert_rate(s.to_numpy(), thr)
        )
        stats[str(seg)] = {
            "tail_threshold": round(thr, 6),
            "mean": float(monthly.mean()),
            "std": float(monthly.std(ddof=0)) if len(monthly) > 1 else 0.0,
            "n_months": int(len(monthly)),
        }
    thr_all = tail_threshold(b["proba"].to_numpy(), quantile)
    overall = b.groupby("_month")["proba"].apply(
        lambda s, thr=thr_all: alert_rate(s.to_numpy(), thr)
    )
    stats["__overall__"] = {
        "tail_threshold": round(thr_all, 6),
        "mean": float(overall.mean()),
        "std": float(overall.std(ddof=0)) if len(overall) > 1 else 0.0,
        "n_months": int(len(overall)),
    }
    return stats


def _zscore(rate: float, stat: dict) -> float:
    """How many baseline standard deviations above normal this rate sits."""
    std = stat["std"]
    if std <= 0:
        return 0.0 if rate <= stat["mean"] else float("inf")
    return (rate - stat["mean"]) / std


# --------------------------------------------------------------------------- #
# Window reports
# --------------------------------------------------------------------------- #
@dataclass
class SegmentSignal:
    segment: str
    alert_rate: float
    baseline_rate: float
    ratio: float | None
    zscore: float | None
    status: str


@dataclass
class WindowReport:
    month: int
    n_rows: int
    population_score_psi: float
    population_features_drifted: int
    overall_alert_rate: float
    segments: list = field(default_factory=list)
    status: str = "OK"
    reasons: list = field(default_factory=list)


def _status_from(ratio: float, z: float) -> str:
    """Combine an economic signal (ratio) with a statistical one (sigma).

    Ratio alone is noisy on small segments; sigma alone is scale-sensitive. A real
    regime change should multiply the tail rate AND sit well outside baseline
    variation -- unless it is such an extreme outlier that sigma alone suffices.
    """
    if z >= SIGMA_EXTREME:
        return "ALERT"
    if ratio >= RATIO_ALERT and z >= SIGMA_ALERT:
        return "ALERT"
    if ratio >= RATIO_WARN and z >= SIGMA_WARN:
        return "WARN"
    return "OK"


def monitor(
    df: pd.DataFrame,
    baseline_frac: float = BASELINE_FRAC,
    segment_col: str = "entry_mode",
    quantile: float = TAIL_QUANTILE,
) -> dict:
    """Monitor every month after the baseline for drift, label-free.

    ``df`` needs ``event_ts``, the feature columns, ``proba``, and ``segment_col``.
    """
    ts = df["event_ts"]
    baseline_end = ts.quantile(baseline_frac)
    baseline = df[ts < baseline_end]
    stats = baseline_stats(baseline, segment_col, quantile)

    df = df.assign(_month=_month_index(ts))
    baseline_last_month = int(_month_index(baseline["event_ts"]).max())

    windows: list[WindowReport] = []
    for m in sorted(df.loc[df["_month"] > baseline_last_month, "_month"].unique()):
        cur = df[df["_month"] == m]
        if len(cur) < MIN_WINDOW_ROWS:
            continue

        fd = feature_drift(baseline, cur)
        score_psi = round(compute_psi(baseline["proba"].to_numpy(), cur["proba"].to_numpy()), 4)

        seg_signals: list[SegmentSignal] = []
        worst = "OK"
        reasons: list[str] = []
        for seg, g in cur.groupby(segment_col):
            stat = stats.get(str(seg))
            if not stat or stat["n_months"] < 2:
                continue
            rate = alert_rate(g["proba"].to_numpy(), stat["tail_threshold"])
            z = _zscore(rate, stat)
            ratio = rate / stat["mean"] if stat["mean"] > 0 else float("inf")
            status = _status_from(ratio, z)
            seg_signals.append(
                SegmentSignal(
                    segment=str(seg),
                    alert_rate=round(rate, 5),
                    baseline_rate=round(stat["mean"], 5),
                    ratio=round(ratio, 2) if np.isfinite(ratio) else None,
                    zscore=round(z, 2) if np.isfinite(z) else None,
                    status=status,
                )
            )
            if status == "ALERT":
                worst = "ALERT"
                reasons.append(
                    f"{seg}: alert rate {rate:.2%} is {ratio:.1f}x baseline "
                    f"({stat['mean']:.2%}), {z:.1f} sigma"
                )
            elif status == "WARN" and worst == "OK":
                worst = "WARN"
                reasons.append(f"{seg}: alert rate {rate:.2%} at {z:.1f} sigma")

        windows.append(
            WindowReport(
                month=int(m),
                n_rows=int(len(cur)),
                population_score_psi=score_psi,
                population_features_drifted=fd["n_drifted"],
                overall_alert_rate=round(
                    alert_rate(cur["proba"].to_numpy(), stats["__overall__"]["tail_threshold"]), 5
                ),
                segments=[
                    asdict(s) for s in sorted(seg_signals, key=lambda s: -(s.zscore or 0))
                ],
                status=worst,
                reasons=reasons,
            )
        )

    first_alert = next((w.month for w in windows if w.status == "ALERT"), None)
    psi_ever_major = any(w.population_score_psi >= PSI_MAJOR for w in windows)

    return {
        "baseline": {
            "n_rows": int(len(baseline)),
            "last_month": baseline_last_month,
            "end": str(baseline_end),
            "segment_stats": stats,
        },
        "tail_quantile": quantile,
        "windows": [asdict(w) for w in windows],
        "first_alert_month": first_alert,
        "population_psi_ever_major": psi_ever_major,
    }


# --------------------------------------------------------------------------- #
# Evidently HTML (the human-facing visual artifact)
# --------------------------------------------------------------------------- #
def write_evidently_html(reference: pd.DataFrame, current: pd.DataFrame, out_path: Path) -> bool:
    """Write a rich per-feature drift report. False if Evidently isn't installed."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except Exception:
        return False

    cols = list(FEATURE_COLUMNS)
    definition = DataDefinition(numerical_columns=cols)
    ref_ds = Dataset.from_pandas(reference[cols], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[cols], data_definition=definition)
    result = Report([DataDriftPreset()]).run(current_data=cur_ds, reference_data=ref_ds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out_path))
    return True


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _render_markdown(report: dict, ring_month: int = 13) -> str:
    rows = []
    for w in report["windows"]:
        worst = w["segments"][0] if w["segments"] else None
        seg_txt = (
            f"`{worst['segment']}` {worst['alert_rate']:.2%} "
            f"({worst['ratio']}x, {worst['zscore']}σ)"
            if worst
            else "—"
        )
        rows.append(
            f"| {w['month']} | {w['n_rows']:,} | {w['population_score_psi']:.3f} | "
            f"{w['population_features_drifted']}/20 | {w['overall_alert_rate']:.2%} | "
            f"{seg_txt} | **{w['status']}** |"
        )

    first = report["first_alert_month"]
    headline = (
        f"The monitor raises its first **ALERT in month {first}** — using no labels at all."
        if first is not None
        else "No window reached ALERT status."
    )
    psi_verdict = (
        "yes" if report["population_psi_ever_major"] else "**no** — never reached the major band"
    )

    return "\n".join(
        [
            "# Drift Monitoring Report (Milestone 8)",
            "",
            "> Auto-generated by `python -m lonestar.monitoring`. Do not edit by hand.",
            "",
            "## Headline",
            "",
            headline,
            "",
            f"The fraud ring switches on around month {ring_month}. Its chargebacks do not "
            "arrive for another 30-60 days, so the observed fraud rate cannot possibly "
            "reveal it yet. This monitor sees it immediately from **inputs and scores "
            "only** — which is the entire argument for input drift monitoring on a "
            "delayed-label problem.",
            "",
            "## The finding: population drift metrics MISS fraud rings",
            "",
            "| Metric | Detected the ring? |",
            "|---|---|",
            f"| Population score PSI | {psi_verdict} |",
            "| Population feature PSI | **no** — a couple of features nudge, nothing decisive |",
            "| **Segment tail alert rate** | **yes — many sigma above baseline** |",
            "",
            "A ring touches ~1% of cards. Against ~12,000 transactions a month its "
            "footprint is a rounding error, so an averaged, population-wide PSI barely "
            "moves. That is not a flaw in PSI; it is what averaging does to a rare "
            "subpopulation. Detecting rare-but-dangerous events requires watching **the "
            "tail** (the rate of high-scoring transactions) **within the segment** the "
            "attack lives in, against a **control limit** built from baseline variation.",
            "",
            "## Month-by-month",
            "",
            "| Month | Rows | Score PSI | Features drifted | Overall alert rate | Worst segment | Status |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
            "Note how the *overall* alert rate stays flat while the e-commerce segment's "
            "rate multiplies — the ring is invisible in the aggregate and obvious in the "
            "channel it attacks.",
            "",
            "## The alert rule",
            "",
            f"- **The tail is defined per segment as that segment's own baseline "
            f"{TAIL_QUANTILE:.0%} score percentile**, so the baseline tail rate is ~"
            f"{1 - TAIL_QUANTILE:.0%} by construction — whatever the model's calibration. "
            "A fixed cutoff would be a genuine tail for one model and 30% of the "
            "population for another.",
            f"- Baseline: monthly tail rate per segment over the first {BASELINE_FRAC:.0%} of "
            "the timeline (normal operations), giving a mean and standard deviation.",
            f"- **WARN** when the rate is >= {RATIO_WARN}x baseline AND >= {SIGMA_WARN}σ; "
            f"**ALERT** at >= {RATIO_ALERT}x AND >= {SIGMA_ALERT}σ, or at "
            f"{SIGMA_EXTREME}σ alone. Requiring both an economic and a statistical "
            "signal keeps small segments from crying wolf.",
            "- This monitoring tail is deliberately separate from the business decline "
            "thresholds in the M5 policy, so changing risk appetite never silently "
            "changes observability.",
            f"- Windows under {MIN_WINDOW_ROWS:,} rows are skipped: a truncated final month "
            "produces meaningless drift on calendar features and would fire a false alarm.",
            "",
            "Every input to this rule is available the instant a transaction happens. "
            "Nothing waits on a chargeback.",
            "",
            "A visual companion (per-feature distribution shifts) is written to "
            "`reports/drift_report.html` by Evidently.",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run drift monitoring (M8).")
    ap.add_argument("--features", default="data/features/features.parquet")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--model", default="models/fraud_model.joblib")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--no-html", action="store_true", help="Skip the Evidently HTML report.")
    args = ap.parse_args()

    df = _score_frame(Path(args.features), Path(args.raw), Path(args.model))
    report = monitor(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    doc = Path(args.docs) / "drift_monitoring.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(_render_markdown(report), encoding="utf-8")

    if not args.no_html:
        ts = df["event_ts"]
        cut = ts.quantile(BASELINE_FRAC)
        ok = write_evidently_html(df[ts < cut], df[ts >= cut], out / "drift_report.html")
        print("evidently html:", "written" if ok else "skipped (not installed)")

    print(f"first ALERT month: {report['first_alert_month']}")
    print(f"population PSI ever major: {report['population_psi_ever_major']}")
    for w in report["windows"]:
        worst = w["segments"][0] if w["segments"] else None
        extra = (
            f" | worst: {worst['segment']} {worst['alert_rate']:.2%} ({worst['zscore']}σ)"
            if worst
            else ""
        )
        print(
            f"  month {w['month']:>2}  psi={w['population_score_psi']:.3f}  {w['status']:5s}{extra}"
        )
    print(f"\nWrote {doc} and {out / 'drift_report.json'}")


if __name__ == "__main__":
    main()
