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

1. **Alert rate, not mean score.** Track the share of transactions scoring above a
   monitoring threshold (0.5). A ring pushes a few transactions to high scores; the
   mean is unchanged, the tail rate multiplies.
2. **Per segment.** Rates are computed per entry mode. The attack lives in
   e-commerce, and monitoring that channel separately turns a rounding error into
   a 3-4x jump.
3. **Control limits from baseline variation.** The baseline period yields a mean
   and standard deviation of the *monthly* alert rate per segment. WARN at 3σ,
   ALERT at 5σ. Using month-to-month variation gives an honest sense of normal
   fluctuation rather than an arbitrary fixed cutoff.
4. **Population PSI is kept and reported** — not as the alert trigger, but as
   evidence in the report that the aggregate view stayed green. The negative result
   is part of the finding.
5. **Minimum window size.** Windows under 2,000 rows are skipped, so a truncated
   final month cannot fire a false alarm on day-of-week.
6. **PSI is implemented natively**; Evidently is used only for the HTML visual.
   Alerting logic should not depend on a third-party dict schema that changes
   between releases, and native PSI is trivially unit-testable.

## Consequences

- On the CI-scale dataset the monitor raises its first ALERT in **month 13**, the
  month the ring becomes visible, at **~7σ on the ECOM segment** — while population
  PSI never leaves the stable band. That is 30-60 days before the chargebacks that
  would have revealed it.
- The entire alerting path is **label-free** (`test_monitoring_path_is_label_free`
  drops the label column and asserts identical alerts), which is mandatory given
  the delay.
- Thresholds are conventions, not laws: 3σ/5σ and a 0.5 monitoring cutoff are
  starting points a risk team would tune against their tolerance for false pages.
- The monitoring threshold is deliberately separate from the M5 business decline
  thresholds, so changing risk appetite does not silently change observability.
