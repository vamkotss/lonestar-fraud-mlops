# Planted Leakage Traps (spoiler)

> This file documents the leaks **on purpose**. In a real project you would find
> these in the Milestone-2 audit, not read them here. They exist so the audit has
> something real to catch, and so "why can't you random-split fraud data?" has a
> concrete, demonstrable answer.

Three columns in `transactions` look like ordinary operational fields but each
secretly encodes the label. Each is a **different archetype** — the audit must
catch three different failure modes, not one.

| Column | Archetype | Why it leaks | How the audit catches it |
|---|---|---|---|
| `dispute_reason_code` | Post-outcome field | Only assigned once a chargeback is filed — strictly *after* the label exists. Present iff fraud. | Perfect class separation (agreement / AUC ≈ 1.0). Any single feature at AUC 1.0 is a leak until proven otherwise. |
| `merchant_fraud_rate_lifetime` | Look-ahead target encoding | The merchant's fraud rate over its **whole** history (future included), stamped on every row — so a January row already "knows" December's fraud. | The value is **constant within `merchant_id`** — impossible for a point-in-time-correct feature. Recompute it as-of and the numbers move. |
| `card_txn_count_next_24h` | Temporal look-ahead window | Counts the card's transactions in the **next** 24h. Uses events after `t`. | Recompute the true forward count and it matches; the honest analogue is a *past* 24h window. |

## The lesson these encode

A random train/test split scatters a card's or merchant's future across both
sides, so all three leaks quietly inflate offline metrics — the model looks
brilliant and dies in production. The fix, enforced structurally by the separate
label table and its `reported_at`, is **temporal splitting** plus point-in-time
(`as_of`) feature computation. Milestone 2 removes these three, re-scores, and
documents the (healthy) drop in offline AUC.

## Audit outcome (Milestone 2 — DONE)

Run `python -m lonestar.audit --data data/raw`. On the CI-scale slice, all three
detectors return **LEAK** and the temporal (past → future) split shows the damage:

| | Test AUC |
|---|---|
| Logistic regression **with** the three leaks | **≈ 1.00** (a fantasy) |
| Same model **without** them (honest features only) | **≈ 0.79** |
| **Inflation attributable purely to leakage** | **≈ +0.21 AUC** |

The full write-up is regenerated each run at
[`docs/leakage_audit_report.md`](leakage_audit_report.md), with EDA context in
[`docs/eda_findings.md`](eda_findings.md) and figures under `reports/figures/`.
Key nuance: a naive **univariate AUC screen catches only `dispute_reason_code`** —
the two look-ahead leaks stay invisible to it, which is exactly why the audit
uses a **structural detector per archetype** rather than one blunt test.
