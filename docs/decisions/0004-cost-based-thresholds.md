# ADR 0004 — Cost-based decision thresholds (global + per-segment)

## Context

The model outputs a probability; production needs an approve/decline decision. The
default 0.5 cutoff optimises the *error count*, which is the wrong objective for
fraud: approving a fraud loses the full transaction amount, while declining a good
customer loses only a little margin plus goodwill. The two mistakes are wildly
asymmetric in dollars, and fraud rates differ several-fold across channels (ECOM
vs CHIP). A single 0.5 threshold ignores both facts.

## Decision

- Define an explicit **cost model**: false negative = transaction amount; false
  positive = a fixed friction charge + a small fraction of the amount.
- Choose the threshold that **minimises expected dollars**, searched over a grid
  adapted to the score distribution.
- Tune **one threshold per segment** (entry mode), with the global optimum as the
  fallback for segments lacking both classes.
- Tune on the **training window**, report savings on the **held-out future** — the
  same temporal discipline as the model itself, so the threshold is not fit to the
  data it is graded on.
- Persist a servable `decision_policy.json` (cost params + global + per-segment
  thresholds) that the scoring service (M6) loads alongside the model.

## Consequences

- On the test window, per-segment thresholds cut total cost below the naive 0.5
  **and** lower the false-decline rate — fewer good customers declined and less
  money lost at once. See `docs/economics_memo.md` (regenerated each run).
- Where the optimal threshold lands depends on model calibration. Our
  imbalance-weighted model already scores fraud high, so the dollar-optimal point
  sits slightly *above* 0.5 (trading a little fraud capture for far fewer false
  declines). The procedure is general; the exact number is model-specific.
- The policy is a small JSON artifact, decoupled from the model binary, so the
  cost assumptions can be re-tuned without retraining.
- A natural extension (not built here) is a three-way approve / review / decline
  policy with an analyst-time cost for the review band.
