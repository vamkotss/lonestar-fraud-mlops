# ADR 0003 — Point-in-time feature computation (backward-only, as-of labels)

## Context

Milestone 2 proved that three features leaked because they used information from
after the transaction. Building real features now means the opposite discipline:
for a transaction at time `t`, a feature may use only data with timestamp `< t`.
Getting this wrong is the most common way a fraud model looks great offline and
fails in production, so the correctness has to be *structural and testable*, not a
convention we hope to remember.

## Options

1. **Compute aggregates with pandas groupby over the whole frame.** Simple, but a
   plain `groupby(...).transform("mean")` averages over the entire history —
   including the future. This is precisely the M2 Leak B pattern. Rejected.
2. **Per-entity sorted windows via `searchsorted`.** Sort by (entity, time); for
   each row, use binary search to count/sum only the strictly-earlier rows inside
   a backward window. O(n log n), vectorised per entity block, and correct by
   construction. Chosen.

For the label: censor by `reported_at`. A row counts as fraud at `as_of` only if
its chargeback was reported by then — the honest labelling M2 deferred.

## Decision

- All aggregate features are **backward-only** windows computed with `searchsorted`
  over per-entity sorted blocks. Ties at the exact same timestamp are excluded
  (`side="left"`), so a row never counts itself.
- `merchant_fraud_rate_asof` uses chargebacks **reported** by `t` over transactions
  **seen** by `t`, smoothed by a Beta(1, 650) prior so a fresh merchant is neutral
  rather than 0/0.
- Correctness is enforced by a **future-proof test**: truncate the future,
  recompute, and require the surviving rows to be identical.
- A single `FEATURE_COLUMNS` spec is the source of truth so training and serving
  compute the identical set (the seed of "feature-store-lite", finished in M7).

## Consequences

- The offline AUC drops from the leaky ≈ 1.0 to an honest ≈ 0.75, which is the
  point.
- The as-of merchant-risk feature is *inverted* on the drift window because the
  label delay hides new attackers. We keep it (a tree model in M4 is not misled by
  the sign) and document it as the finding that motivates drift monitoring (M8) and
  retraining (M9). See `docs/feature_pipeline.md`.
- Slightly more code than a naive groupby, paid back by a leakage guarantee a
  reviewer can run in one command.
