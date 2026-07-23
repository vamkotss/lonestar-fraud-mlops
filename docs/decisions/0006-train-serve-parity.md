# ADR 0006 — Train/serve parity via a shared decision core

## Context

The project now has two scoring paths: the online service (M6) for real-time
single transactions, and the batch scorer (M7) for bulk runs. If these two paths
computed the segment, threshold, or decision even slightly differently, they would
return different decisions for the same transaction — serving skew, one of the
most damaging and hardest-to-notice production ML failures.

## Decision

1. **One shared decision core, `lonestar.scoring`.** Segment recovery from the
   entry one-hots, per-segment threshold lookup, and the decline rule live in one
   module. Both the online route and the batch scorer import and call it. There is
   no second implementation to drift.

2. **Both paths use the same artifacts.** The same `fraud_model.joblib`,
   `model_card.json`, and `decision_policy.json` drive both, so the model and the
   thresholds are identical, not merely similar.

3. **One feature definition, `lonestar.features`.** Training and scoring build
   features from the same `FEATURE_COLUMNS` and the same functions. The online
   service additionally validates each request against the card's `feature_columns`
   so a schema mismatch fails loudly.

4. **Parity is tested, not assumed.** `test_online_batch_parity` scores the same
   transactions through the live API and the batch job and asserts identical
   probabilities (to rounding) and identical decisions.

## Consequences

- Online and batch decisions are provably consistent; a divergence would fail CI.
- Refactors are safe: changing the decision rule in one place changes both paths,
  and the parity test guards the invariant.
- The batch scorer is a thin, vectorised wrapper around the shared core, so it
  stays fast without duplicating logic.
- Adding a third path later (e.g. a streaming scorer in M6-of-P6) would reuse the
  same core and inherit the same guarantee.
