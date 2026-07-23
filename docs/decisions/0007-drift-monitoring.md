# ADR 0007 — Drift monitoring for a rare-event problem

## Context

Chargebacks arrive 30-60 days after the transaction (ADR 0003). The observed fraud
rate is therefore a lagging indicator: by the time it moves, the attack has already
been paid for. Monitoring has to work on what is available immediately — inputs and
model scores — and it has to catch a fraud ring that touches roughly 1% of cards.

## What we tried first, and why it failed

The obvious design is a population-level drift report: PSI on every feature and on
the score distribution, reference window versus current month. We built that first.
**It does not detect the ring.** Against ~12,000 transactions a month, a ring
touching ~1% of cards barely moves an averaged distribution; population score PSI
peaked around 0.06, nowhere near the 0.25 "major" band, in the very months the ring
was active. The only features that did drift were calendar artifacts of a truncated
final month — a false alarm, in the wrong month, for the wrong reason.

This is not a flaw in PSI. It is what averaging does to a rare subpopulation, and
it is a trap worth knowing about: a monitoring dashboard full of green population
metrics can sit on top of an active attack.

## Decision

Alert on the **tail, per segment, against a control limit**:

1. **Alert rate, not mean score.** Track the share of transactions in the score
   tail. A ring pushes a few transactions to high scores; the mean is unchanged,
   the tail rate multiplies.

2. **The tail is a quantile of each segment's OWN baseline, not a fixed cutoff.**
   This was learned the hard way. The first version used a fixed 0.5 monitoring
   threshold; that is a genuine tail for the CI-scale model but captures 15-30% of
   the population for the imbalance-weighted full-scale model. At that point the
   "tail rate" measures the bulk, its baseline variance explodes, and nothing ever
   trips the control limit -- the full-scale monitor raised **no alert at all**.
   Defining the tail as the segment's baseline 99th percentile pins the baseline
   rate at ~1% *by construction*, whatever the model's calibration.
   `test_alerts_are_independent_of_model_calibration` applies a monotonic squash to
   the scores and asserts the alerts are unchanged.
3. **Per segment.** Rates are computed per entry mode. The attack lives in
   e-commerce, and monitoring that channel separately turns a rounding error into
   a 3-4x jump.
4. **Control limits combine an economic and a statistical signal.** WARN at >=1.5x
   baseline AND >=2σ; ALERT at >=2x AND >=3σ (or >=5σ alone). Ratio alone is noisy
   on small segments; sigma alone is scale-sensitive. Requiring both keeps small
   channels from crying wolf while still catching a real multiplication of risk.
5. **Population PSI is kept and reported** — not as the alert trigger, but as
   evidence in the report that the aggregate view stayed green. The negative result
   is part of the finding.
6. **Minimum window size.** Windows under 2,000 rows are skipped, so a truncated
   final month cannot fire a false alarm on day-of-week.
7. **PSI is implemented natively**; Evidently is used only for the HTML visual.
   Alerting logic should not depend on a third-party dict schema that changes
   between releases, and native PSI is trivially unit-testable.

## Consequences

- The monitor raises its first ALERT in **month 13**, the month the ring becomes
  visible, at **~5σ / 3.7x baseline on the ECOM segment**, and holds an elevated
  WARN/ALERT state through the attack — while population PSI never leaves the
  stable band. That is 30-60 days before the chargebacks that would have revealed
  it.
- The scale lesson generalises: **a monitoring rule tuned on a small sample can be
  silently wrong at production scale.** Both bugs in this project's back half (the
  in-sample thresholds of ADR 0004 and the fixed cutoff here) only appeared on the
  full 5M-row run.
- The entire alerting path is **label-free** (`test_monitoring_path_is_label_free`
  drops the label column and asserts identical alerts), which is mandatory given
  the delay.
- Thresholds are conventions, not laws: 3σ/5σ and a 0.5 monitoring cutoff are
  starting points a risk team would tune against their tolerance for false pages.
- The monitoring threshold is deliberately separate from the M5 business decline
  thresholds, so changing risk appetite does not silently change observability.
