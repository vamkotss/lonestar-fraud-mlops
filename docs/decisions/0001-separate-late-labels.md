# ADR 0001 — Separate late-arriving labels from transactions

## Context

Fraud labels (chargebacks) do not exist at transaction time. They materialize 30–60
days later when a customer disputes a charge. A naive schema stores `is_fraud` as a
column on the transaction row, filled in later. That design silently invites the
single most damaging error in fraud modeling: training on a label that, on the date
you're predicting, had not yet arrived — a form of temporal leakage.

## Options

1. **`is_fraud` nullable column on `transactions`.** Simple, but every query and
   feature computation must *remember* to respect timing, and nothing enforces it.
2. **Separate `chargeback_labels` table with a `reported_at` timestamp.** The label's
   arrival time is first-class data. Point-in-time-correct features and temporal
   training splits fall out naturally: you join labels only where
   `reported_at <= as_of`.

## Decision

Option 2. Labels live in their own table with `reported_at`.

## Consequences

- Point-in-time correctness becomes a *structural* property, not a discipline you
  hope to remember. The M3 feature pipeline and M4 temporal splits both key off
  `reported_at`.
- Slightly more join complexity, paid back many times over in leakage safety.
- Directly answers the interview question this project is built around: *"Why can't
  you random-split fraud data?"* — because the label horizon moves, and a random
  split lets tomorrow's knowledge train today's model.
