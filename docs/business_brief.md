# Business Brief — Lonestar Card Services Fraud Detection

## Context

Lonestar Card Services is a fictional regional card issuer with roughly 800,000
active cards and about 90 million transactions a year (this project simulates a
representative slice). Its current fraud defense is a rules engine with 200+
hand-written thresholds that nobody dares edit — every change risks breaking
something no one fully understands.

## The pain point — two losses that pull in opposite directions

The core tension of this project, and the reason it is a genuine business problem
and not a modeling exercise:

- **Undetected fraud** costs roughly **$0.9M/year** (modeled). Every fraudulent
  transaction that slips through is a direct loss.
- **False declines** cost *more*. When the system blocks a good customer at
  checkout, that customer abandons the purchase and sometimes leaves for a
  competitor. The lifetime cost of that lost customer exceeds the fraud it was
  trying to prevent — which matches real-industry economics.

A fraud model that only chases catch-rate makes the false-decline problem worse.
The whole project is built around **minimizing total dollar loss**, not maximizing
a classification metric.

## Stakeholders — and their conflicting incentives (by design)

| Stakeholder | Owns | Wants |
|---|---|---|
| Head of Fraud (sponsor) | The loss numbers | Higher catch rate |
| VP Customer Experience | Decline complaints | Fewer false declines — *directly opposed to the Head of Fraud* |
| On-call fraud ops | Alert queue | Clear, explained alerts they can action fast |
| Model risk officer | Governance sign-off | A model card, documented limitations, a rollback plan |

The opposed incentives are the point: the decision layer has to make a defensible
tradeoff between two stakeholders who want opposite things.

## Business KPIs

- **Total dollar loss** (fraud losses + false-decline costs) — the headline
- Fraud catch rate at a fixed review capacity
- False positive rate by customer segment
- Alert precision@k
- Scoring latency p99 (the model has to be fast enough to sit in the checkout path)

## Success criteria

1. **Leakage-audited features**, models validated on temporal splits, all tracked in MLflow.
2. Decision **thresholds chosen by expected-dollar-loss minimization per segment** — not by F1.
3. An **online scoring API** (<100ms local p99, authenticated) *and* a nightly
   **batch scorer**, provably consistent (same features, same model).
4. **Drift monitoring** that catches an injected new fraud ring appearing at month 14.
5. **Automated challenger/champion retraining** with a promotion gate.
6. A **model card** the risk officer would accept.

## Expected value

In simulation, a target of **≥30% total-cost reduction** versus the rules
baseline, with the dollar math shown in a one-page economics memo.

## Assumptions

- **Chargeback labels arrive 30–60 days late.** This is not a nuisance to work
  around — the entire pipeline is designed around a moving labeled horizon, because
  that is the reality of fraud data.
- Cost-matrix figures are approximations, documented where used.

## Risks

- **Leakage inflating offline results** — mitigated by a deliberate planted-leak
  audit in Milestone 2.
- **Threshold gaming a bad cost matrix** — mitigated by sensitivity analysis.
- **Drift alarms crying wolf** — mitigated by a defined alert budget.
