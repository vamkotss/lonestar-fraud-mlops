"""Seeded transaction + delayed-label generator for Lonestar Card Services.

Design contract (why this exists and what it guarantees):

  * ~5M transaction ROWS in full mode; a small-but-representative slice under
    ``LS_CI_MODE`` that keeps EVERY property (imbalance, delay, drift, all three
    leaks, all the mess) so CI proves the same invariants in seconds.
  * ~0.15% base fraud -- extreme class imbalance, the whole point of the project.
  * Labels live in a SEPARATE table (``chargeback_labels``) with a ``reported_at``
    that lands 30-60 days after the transaction. Fraud is NOT a column on the
    transactions table -- only the three planted leaks pretend to be.
  * A fraud RING switches on at month 14: new high-risk merchants, e-commerce
    entry, foreign geography, a specific spend band. That distribution shift is
    the concept drift Milestone 8 must detect.
  * Realistic mess: 4% missing MCCs, chaotic merchant descriptors, duplicate
    auth/capture pairs, timezone-naive timestamps.

Run it:  ``python -m lonestar.generation --out data/raw``  (respects LS_CI_MODE / LS_SEED)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lonestar import _catalog, _leaks

# Anchor date for the simulated history. Month 1 == 2024-01.
_START = pd.Timestamp("2024-01-01")
_DEFAULT_SEED = 20260721  # matches LS_SEED in .env.example


# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenConfig:
    """All generation knobs. ``from_env`` picks full vs CI-scale from LS_CI_MODE."""

    n_purchases: int          # logical purchases (each may spawn an auth + capture row)
    months: int               # length of simulated history
    ring_onset_month: int     # month the fraud ring switches on
    base_fraud_rate: float    # baseline fraud share of purchases
    ring_extra_rate: float    # additional fraud share applied to onset-window purchases
    ring_card_frac: float     # fraction of cards belonging to the ring
    capture_rate: float       # P(a purchase also produces a capture row)
    partial_capture_rate: float  # P(capture amount differs from auth | captured)
    missing_mcc_rate: float   # fraction of rows with a NULL mcc
    label_delay_min: int      # min chargeback delay (days)
    label_delay_mode: int     # triangular peak (days)
    label_delay_max: int      # max chargeback delay (days)
    avg_purchases_per_card: int
    seed: int
    ci_mode: bool

    @classmethod
    def from_env(cls) -> GenConfig:
        ci = os.environ.get("LS_CI_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
        seed = int(os.environ.get("LS_SEED", _DEFAULT_SEED))
        # Full: 2.7M purchases * ~1.87 (capture rate) ~= 5.0M transaction rows.
        # CI:   120K purchases keeps 0.15% statistically solid and a visible ring,
        #       yet runs in a few seconds.
        n_purchases = 120_000 if ci else 2_700_000
        return cls(
            n_purchases=n_purchases,
            months=18,
            ring_onset_month=14,
            base_fraud_rate=0.0015,
            ring_extra_rate=0.0040,
            ring_card_frac=0.01,
            capture_rate=0.87,
            partial_capture_rate=0.06,
            missing_mcc_rate=0.04,
            label_delay_min=30,
            label_delay_mode=45,
            label_delay_max=60,
            avg_purchases_per_card=25,
            seed=seed,
            ci_mode=ci,
        )


def _month_index(ts: pd.Series) -> np.ndarray:
    """1-based month index relative to _START (Jan 2024 -> 1)."""
    return ((ts.dt.year - _START.year) * 12 + ts.dt.month).to_numpy()


# ---------------------------------------------------------------------------- #
# Generation
# ---------------------------------------------------------------------------- #
def generate(cfg: GenConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the transactions table and the separate chargeback_labels table.

    Returns ``(transactions, chargeback_labels, manifest)``.
    """
    rng = np.random.default_rng(cfg.seed)
    P = cfg.n_purchases
    cat = _catalog.build_catalog(rng)

    # --- cards & accounts -----------------------------------------------------
    n_cards = max(50, P // cfg.avg_purchases_per_card)
    p_card = rng.integers(0, n_cards, size=P)          # which card made each purchase
    # A designated slice of cards belongs to the ring.
    n_ring_cards = max(1, int(n_cards * cfg.ring_card_frac))
    ring_cards = set(rng.choice(n_cards, size=n_ring_cards, replace=False).tolist())
    ring_card_mask = np.isin(p_card, list(ring_cards))

    # --- timestamps (timezone-naive on purpose) -------------------------------
    end = _START + pd.DateOffset(months=cfg.months)
    span_s = (end - _START).total_seconds()
    offset_s = rng.random(P) * span_s
    p_auth_ts = _START + pd.to_timedelta(offset_s, unit="s")
    p_auth_ts = pd.Series(p_auth_ts).dt.floor("s")     # drop sub-second noise
    p_month = _month_index(p_auth_ts)
    onset_mask = p_month >= cfg.ring_onset_month

    # --- merchant assignment (Zipf-ish popularity) ----------------------------
    n_honest = cat["ring_start"]
    pop = 1.0 / np.arange(1, n_honest + 1)             # honest merchants only, by default
    pop = pop / pop.sum()
    p_merchant = rng.choice(n_honest, size=P, p=pop)

    # --- entry mode / geography / amount (honest defaults) --------------------
    entry_modes = np.array(["CHIP", "CONTACTLESS", "SWIPE", "ECOM", "MANUAL"], dtype=object)
    p_entry = rng.choice(entry_modes, size=P, p=[0.42, 0.28, 0.10, 0.18, 0.02])
    p_country = np.full(P, "US", dtype=object)
    amt_mu = cat["amt_mu"][p_merchant]
    amt_sigma = cat["amt_sigma"][p_merchant]
    p_amount = np.round(rng.lognormal(amt_mu, amt_sigma), 2)
    p_amount = np.clip(p_amount, 1.0, 25_000.0)

    # --- fraud assignment ------------------------------------------------------
    is_fraud = np.zeros(P, dtype=bool)
    fraud_type = np.empty(P, dtype=object)
    fraud_type[:] = "NONE"

    # (1) Baseline fraud: an exact count spread uniformly across the whole span.
    n_base = int(round(cfg.base_fraud_rate * P))
    base_idx = rng.choice(P, size=n_base, replace=False)
    is_fraud[base_idx] = True
    fraud_type[base_idx] = "BASELINE"

    # (2) Ring fraud: onset-window purchases on ring cards, re-pointed at ring
    #     merchants with ring-signature features. This is what shifts the feature
    #     distribution (the drift) AND lifts the fraud rate after month 14.
    ring_candidates = np.flatnonzero(onset_mask & ring_card_mask)
    n_ring = int(round(cfg.ring_extra_rate * int(onset_mask.sum())))
    n_ring = min(n_ring, ring_candidates.shape[0])
    ring_idx = rng.choice(ring_candidates, size=n_ring, replace=False)

    is_fraud[ring_idx] = True
    fraud_type[ring_idx] = "RING"                      # RING wins if a row was also base
    # Ring signature -> the drift the monitor should catch.
    ring_merchant_choices = np.arange(cat["ring_start"], cat["ring_start"] + 4)
    p_merchant[ring_idx] = rng.choice(ring_merchant_choices, size=n_ring)
    p_entry[ring_idx] = "ECOM"
    p_country[ring_idx] = rng.choice(
        np.array(["US", "GB", "RO", "NG"], dtype=object), size=n_ring, p=[0.25, 0.30, 0.25, 0.20]
    )
    p_amount[ring_idx] = np.round(rng.uniform(200.0, 600.0, size=n_ring), 2)

    # Recompute merchant-derived fields after ring re-pointing.
    p_mcc = cat["mcc"][p_merchant].astype(np.int64)
    p_merchant_id = cat["merchant_id"][p_merchant]

    # --- messy merchant descriptors (many names per merchant_id) ---------------
    nv = cat["n_variants"]
    variant_choice = rng.integers(0, nv, size=P)
    flat_variants = cat["name_variants"].reshape(-1)
    p_merchant_name = flat_variants[p_merchant * nv + variant_choice]

    # --- purchase-grain frame --------------------------------------------------
    auth_code = np.array([f"A{cfg.seed % 100:02d}{i:09d}" for i in range(P)], dtype=object)
    purchases = pd.DataFrame(
        {
            "auth_code": auth_code,
            "card_id": np.array([f"C{c:07d}" for c in p_card], dtype=object),
            "account_id": np.array([f"AC{c // 2:07d}" for c in p_card], dtype=object),
            "merchant_id": p_merchant_id,
            "merchant_name": p_merchant_name,
            "mcc": p_mcc,
            "amount": p_amount,
            "currency": "USD",
            "entry_mode": p_entry,
            "pos_country": p_country,
            "event_ts": p_auth_ts.to_numpy(),
            "response_code": "APPROVED",
            "_is_fraud": is_fraud,
            "_fraud_type": fraud_type,
            "_month": p_month,
        }
    )

    # --- explode into auth + capture rows (the duplicate-pair mess) ------------
    auth_rows = purchases.copy()
    auth_rows["event_type"] = "AUTH"

    captured = rng.random(P) < cfg.capture_rate
    cap = purchases.loc[captured].copy()
    cap["event_type"] = "CAPTURE"
    # Capture lands 1h-72h after auth; timezone-naive throughout.
    lag_s = rng.integers(3600, 72 * 3600, size=len(cap))
    cap["event_ts"] = cap["event_ts"].to_numpy() + pd.to_timedelta(lag_s, unit="s")
    # Some captures settle for a slightly different amount (tips / partials).
    partial = rng.random(len(cap)) < cfg.partial_capture_rate
    factor = np.where(partial, rng.uniform(0.80, 1.20, size=len(cap)), 1.0)
    cap["amount"] = np.round(cap["amount"].to_numpy() * factor, 2)

    tx = pd.concat([auth_rows, cap], ignore_index=True)
    # A stable, unique per-row id (auth and capture are different rows).
    tx = tx.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    tx.insert(0, "transaction_id", [f"T{cfg.seed % 100:02d}{i:010d}" for i in range(len(tx))])

    # --- planted leaks (computed on the full row set) --------------------------
    is_fraud_row = tx["_is_fraud"].to_numpy()
    tx["dispute_reason_code"] = _leaks.add_leak_a_dispute_code(rng, is_fraud_row)
    tx["merchant_fraud_rate_lifetime"] = _leaks.add_leak_b_merchant_lifetime_rate(
        tx["merchant_id"].to_numpy(), is_fraud_row
    )
    tx["card_txn_count_next_24h"] = _leaks.add_leak_c_next_24h_count(
        tx["card_id"].to_numpy(), tx["event_ts"].to_numpy()
    )

    # --- missing MCCs (4%) -----------------------------------------------------
    tx["mcc"] = tx["mcc"].astype("Int64")
    miss = rng.random(len(tx)) < cfg.missing_mcc_rate
    tx.loc[miss, "mcc"] = pd.NA

    # --- timezone-naive timestamps (explicit) ----------------------------------
    tx["event_ts"] = pd.to_datetime(tx["event_ts"]).astype("datetime64[us]")

    # ---------------------------------------------------------------------- #
    # chargeback_labels: one row per FRAUD purchase, reported 30-60 days later.
    # This is the ONLY place fraud is stated as a label -- and it arrives late.
    # ---------------------------------------------------------------------- #
    fraud_pur = purchases[purchases["_is_fraud"]].copy()
    n_f = len(fraud_pur)
    delay_days = np.round(
        rng.triangular(cfg.label_delay_min, cfg.label_delay_mode, cfg.label_delay_max, size=n_f)
    ).astype(int)
    reported_at = fraud_pur["event_ts"].to_numpy() + pd.to_timedelta(delay_days, unit="D")

    # Map each fraud purchase to its AUTH transaction_id for convenience.
    auth_lookup = (
        tx[tx["event_type"] == "AUTH"][["auth_code", "transaction_id"]]
        .drop_duplicates("auth_code")
        .set_index("auth_code")["transaction_id"]
    )
    labels = pd.DataFrame(
        {
            "auth_code": fraud_pur["auth_code"].to_numpy(),
            "transaction_id": auth_lookup.reindex(fraud_pur["auth_code"].to_numpy()).to_numpy(),
            "is_fraud": True,
            "fraud_type": fraud_pur["_fraud_type"].to_numpy(),
            "chargeback_amount": fraud_pur["amount"].to_numpy(),
            "reason_code": _leaks._FRAUD_REASON_CODES[
                rng.integers(0, len(_leaks._FRAUD_REASON_CODES), size=n_f)
            ],
            "event_ts": pd.to_datetime(fraud_pur["event_ts"].to_numpy()).astype("datetime64[us]"),
            "reported_at": pd.to_datetime(reported_at).astype("datetime64[us]"),
        }
    )

    # --- drop internal helper cols from the shipped transactions table --------
    tx = tx.drop(columns=["_is_fraud", "_fraud_type", "_month"])

    # --- manifest (reproducibility + honest summary) ---------------------------
    overall_rate = n_f / P
    pre = purchases["_month"] < cfg.ring_onset_month
    post = ~pre
    manifest = {
        "seed": cfg.seed,
        "ci_mode": cfg.ci_mode,
        "n_purchases": int(P),
        "n_transaction_rows": int(len(tx)),
        "n_fraud_purchases": int(n_f),
        "overall_fraud_rate": round(overall_rate, 6),
        "pre_onset_fraud_rate": round(float(purchases.loc[pre, "_is_fraud"].mean()), 6),
        "post_onset_fraud_rate": round(float(purchases.loc[post, "_is_fraud"].mean()), 6),
        "ring_onset_month": cfg.ring_onset_month,
        "n_ring_fraud": int((fraud_pur["_fraud_type"] == "RING").sum()),
        "missing_mcc_rate": round(float(tx["mcc"].isna().mean()), 4),
        "leak_columns": _leaks.LEAK_COLUMNS,
        "config": asdict(cfg),
    }
    return tx, labels, manifest


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
def _write(tx: pd.DataFrame, labels: pd.DataFrame, manifest: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(out / "transactions.parquet", index=False)
    labels.to_parquet(out / "chargeback_labels.parquet", index=False)
    (out / "_generation_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # A tiny, git-committable CSV peek for the README.
    sample_dir = Path("data/samples")
    if sample_dir.exists():
        tx.head(200).to_csv(sample_dir / "transactions_sample.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Lonestar fraud transactions + labels.")
    parser.add_argument("--out", default="data/raw", help="output directory for parquet files")
    parser.add_argument("--purchases", type=int, default=None, help="override purchase count")
    parser.add_argument("--seed", type=int, default=None, help="override LS_SEED")
    args = parser.parse_args(argv)

    cfg = GenConfig.from_env()
    if args.purchases is not None:
        cfg = GenConfig(**{**asdict(cfg), "n_purchases": args.purchases})
    if args.seed is not None:
        cfg = GenConfig(**{**asdict(cfg), "seed": args.seed})

    tx, labels, manifest = generate(cfg)
    _write(tx, labels, manifest, Path(args.out))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
