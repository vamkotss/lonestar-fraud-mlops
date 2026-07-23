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
- Tune on a **held-out calibration window** and report savings on a later **test
  window**, using a three-way temporal split (see below).
- Persist a servable `decision_policy.json` (cost params + global + per-segment
  thresholds) that the scoring service (M6) loads alongside the model.

## The three-way split (and the bug that forced it)

The first version tuned thresholds on the earliest 70% of the timeline — which sat
**entirely inside the model's own 80% training window**. An overfit model scores its
training fraud near 1.0, so the tuned threshold rested on memorisation rather than
signal. At CI scale this hid the problem (the model overfits enough to look
discriminative); at full 5M-row scale it surfaced dramatically: with less relative
overfitting the fake signal vanished, every threshold optimised to the
never-decline sentinel, and the batch scorer declined **0 of 5,049,290**
transactions — the decision layer silently switched off.

The fix is a strict three-way temporal split:

| Window | Fraction | Purpose |
|---|---|---|
| Train | `[0, TRAIN_FRAC)` | the model fits here (M4) |
| **Calibration** | `[TRAIN_FRAC, 0.90)` | **thresholds tuned here — held out from the model** |
| Test | `[0.90, 1.0)` | economics reported here — unseen by both |

`TRAIN_FRAC` is imported from `lonestar.modeling`, so the calibration window can
never drift back into the model's training data. `test_three_way_split_calibration_is_held_out`
and `test_policy_is_not_degenerate` guard both properties.

## Consequences

- On the held-out test window, per-segment thresholds cut total cost below the
  naive 0.5 **and** cut the false-decline rate by roughly 4x (0.54% → 0.14%) while
  catching ~92% of fraud dollars. See `docs/economics_memo.md`.
- The per-segment result is economically interesting rather than a uniform
  tightening: the **ECOM** channel (which the fraud ring attacks, and where fraud
  is therefore predictable) gets an aggressive low threshold, while card-present
  channels — whose fraud is unpredictable baseline noise — optimise to
  **never decline**, because intervening there costs money and catches nothing the
  model can see. The cost model asks not just *how likely is fraud* but *where is
  intervening worth the money*.
- Where the optimal threshold lands depends on model calibration and on the
  regime; the procedure generalises, the exact numbers do not.
- The policy is a small JSON artifact, decoupled from the model binary, so the
  cost assumptions can be re-tuned without retraining.
- A natural extension (not built here) is a three-way approve / review / decline
  policy with an analyst-time cost for the review band.
