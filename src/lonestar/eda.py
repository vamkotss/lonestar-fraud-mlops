"""Milestone 2 -- exploratory data analysis (EDA).

EDA earns its place in this project by answering three questions a reviewer will
ask before trusting any model:

  1. How rare is fraud, really?  (class imbalance -> why accuracy is useless here)
  2. Is the data stationary?     (the month-14 ring -> why a model decays)
  3. How messy is it?            (missing MCCs, duplicate auth/capture, chaos)

It produces a written findings doc and two figures. The figures are deliberately
plain -- a hiring manager should be able to read the story in five seconds.

Run it:  ``python -m lonestar.eda --data data/raw``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lonestar import _labeling

_RING_ONSET_MONTH = 14  # matches the generator's config
_START = pd.Timestamp("2024-01-01")


def _month_index(ts: pd.Series) -> np.ndarray:
    """Whole months since the dataset start (month 0 = Jan 2024)."""
    return ((ts.dt.year - _START.year) * 12 + (ts.dt.month - _START.month)).to_numpy()


def compute_eda(tx: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Return a dict of the headline EDA findings (all JSON-serialisable)."""
    y = _labeling.attach_row_label(tx, labels).to_numpy()
    month = _month_index(tx["event_ts"])

    # --- 1. class imbalance ---------------------------------------------------
    overall_rate = float(y.mean())

    # --- 2. non-stationarity: fraud rate per month ----------------------------
    frame = pd.DataFrame({"month": month, "fraud": y})
    by_month = frame.groupby("month")["fraud"].mean()
    pre = float(frame.loc[frame.month < _RING_ONSET_MONTH, "fraud"].mean())
    post = float(frame.loc[frame.month >= _RING_ONSET_MONTH, "fraud"].mean())

    # --- ring signature: how the feature mix shifts after onset ---------------
    post_mask = month >= _RING_ONSET_MONTH
    fraud_post = y & post_mask
    fraud_pre = y & ~post_mask

    def _share(mask, condition):
        sub = tx.loc[mask]
        return float(condition(sub).mean()) if len(sub) else float("nan")

    ring_signature = {
        "ecom_share_fraud_pre": _share(fraud_pre, lambda d: d["entry_mode"] == "ECOM"),
        "ecom_share_fraud_post": _share(fraud_post, lambda d: d["entry_mode"] == "ECOM"),
        "foreign_share_fraud_pre": _share(fraud_pre, lambda d: d["pos_country"] != "US"),
        "foreign_share_fraud_post": _share(fraud_post, lambda d: d["pos_country"] != "US"),
    }

    # --- 3. data-quality mess -------------------------------------------------
    mess = {
        "missing_mcc_rate": round(float(tx["mcc"].isna().mean()), 4),
        "names_per_merchant": round(
            float(tx.groupby("merchant_id")["merchant_name"].nunique().mean()), 1
        ),
        "auth_capture_pairs": int((tx["event_type"] == "CAPTURE").sum()),
        "tz_naive": bool(tx["event_ts"].dt.tz is None),
    }

    return {
        "n_rows": int(len(tx)),
        "overall_fraud_rate": round(overall_rate, 6),
        "fraud_rate_by_month": {int(m): round(float(v), 6) for m, v in by_month.items()},
        "pre_onset_fraud_rate": round(pre, 6),
        "post_onset_fraud_rate": round(post, 6),
        "ring_onset_month": _RING_ONSET_MONTH,
        "ring_signature": ring_signature,
        "data_quality": mess,
    }


def write_figures(tx: pd.DataFrame, labels: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render the two story figures. Imports matplotlib lazily (optional dep)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed, works in CI
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    findings = compute_eda(tx, labels)
    written: list[Path] = []

    # Figure 1: monthly fraud rate, with the ring onset marked. THE money chart.
    months = sorted(findings["fraud_rate_by_month"])
    rates = [findings["fraud_rate_by_month"][m] * 100 for m in months]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, rates, marker="o", color="#1f4e79")
    ax.axvline(_RING_ONSET_MONTH, color="#c0392b", linestyle="--", label="fraud ring onset")
    ax.set_title("Fraud rate climbs when the ring switches on (month 14)")
    ax.set_xlabel("months since Jan 2024")
    ax.set_ylabel("fraud rate (%)")
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "fraud_rate_by_month.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    written.append(p1)

    # Figure 2: fraud amount distribution, pre vs post onset (the $200-600 band).
    y = _labeling.attach_row_label(tx, labels).to_numpy()
    month = _month_index(tx["event_ts"])
    amt_pre = tx.loc[y & (month < _RING_ONSET_MONTH), "amount"].clip(upper=1000)
    amt_post = tx.loc[y & (month >= _RING_ONSET_MONTH), "amount"].clip(upper=1000)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1000, 41)
    ax.hist(amt_pre, bins=bins, alpha=0.6, density=True, label="fraud, pre-onset", color="#7f8c8d")
    ax.hist(amt_post, bins=bins, alpha=0.6, density=True, label="fraud, post-onset", color="#c0392b")
    ax.set_title("Post-onset fraud concentrates in the $200-600 band (ring signature)")
    ax.set_xlabel("transaction amount ($, capped at 1000)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "fraud_amount_pre_post.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    written.append(p2)

    return written


def _render_markdown(f: dict) -> str:
    rs = f["ring_signature"]
    dq = f["data_quality"]
    return "\n".join(
        [
            "# EDA Findings (Milestone 2)",
            "",
            "> Auto-generated by `python -m lonestar.eda`. Do not edit by hand.",
            "",
            "## 1. Fraud is rare -- accuracy is a useless metric here",
            "",
            f"- Overall row-level fraud rate: **{f['overall_fraud_rate']:.4%}** "
            f"across **{f['n_rows']:,}** rows.",
            "- A model that predicts 'never fraud' would be ~99.7% accurate and "
            "catch zero fraud. This is why the project scores with AUC and, later, "
            "dollar-weighted cost -- never raw accuracy.",
            "",
            "## 2. The data is NOT stationary -- a fraud ring switches on at month 14",
            "",
            f"- Pre-onset fraud rate: **{f['pre_onset_fraud_rate']:.4%}**",
            f"- Post-onset fraud rate: **{f['post_onset_fraud_rate']:.4%}** "
            f"(~{f['post_onset_fraud_rate'] / max(f['pre_onset_fraud_rate'], 1e-9):.1f}x higher)",
            "- The ring does not just raise the rate; it shifts the *feature mix*:",
            f"  - ECOM share of fraud: {rs['ecom_share_fraud_pre']:.0%} → "
            f"**{rs['ecom_share_fraud_post']:.0%}**",
            f"  - Foreign share of fraud: {rs['foreign_share_fraud_pre']:.0%} → "
            f"**{rs['foreign_share_fraud_post']:.0%}**",
            "- This is the drift the monitoring milestone (M8) must catch. See "
            "`reports/figures/fraud_rate_by_month.png`.",
            "",
            "## 3. The data is messy -- on purpose",
            "",
            f"- Missing MCC codes: **{dq['missing_mcc_rate']:.1%}** of rows.",
            f"- Merchant names are chaotic: ~**{dq['names_per_merchant']}** distinct "
            "descriptors per real merchant (casing, gateway prefixes, store numbers).",
            f"- **{dq['auth_capture_pairs']:,}** capture rows shadow their auth rows "
            "(the duplicate-pair mess a real ledger has).",
            f"- Timestamps are timezone-naive: `{dq['tz_naive']}` -- a deliberate trap "
            "the feature pipeline (M3) must handle explicitly.",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Milestone 2 EDA.")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--no-figures", action="store_true", help="Skip PNG rendering.")
    args = ap.parse_args()

    data = Path(args.data)
    tx = pd.read_parquet(data / "transactions.parquet")
    labels = pd.read_parquet(data / "chargeback_labels.parquet")

    findings = compute_eda(tx, labels)

    out_md = Path(args.out) / "eda_findings.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(findings), encoding="utf-8")

    if not args.no_figures:
        figs = write_figures(tx, labels, Path(args.figures))
        print("Figures:", ", ".join(str(p) for p in figs))

    print(json.dumps(findings, indent=2))
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
