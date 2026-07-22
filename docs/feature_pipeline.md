# Feature Pipeline (Milestone 3)

Point-in-time-correct features: for a transaction at time `t`, **every feature uses
only information that existed strictly before `t`.** This is the honest replacement
for the three leaks caught in Milestone 2.

Build it: `python -m lonestar.features --data data/raw --out data/features`

## The guarantee that matters

`is_future_proof()` deletes the latest 20% of the timeline, recomputes features on
what remains, and confirms the surviving rows are **byte-for-byte identical** to
their values in the full run. If any feature peeked forward, truncating the future
would move those values. It does not — see `test_features_are_future_proof`.

This is the whole point of the milestone: a believable offline score requires that
training never sees the future.

## Feature dictionary

| Feature | Group | Point-in-time guarantee |
|---|---|---|
| `log_amount` | row-static | Known at swipe time. |
| `hour`, `dow`, `is_weekend` | row-static | Calendar of the event itself. |
| `entry_*` (5 one-hots) | row-static | Entry mode is known at authorization. |
| `is_foreign`, `mcc_missing`, `is_capture` | row-static | Known at the event. |
| `card_txn_count_prev_1h / 24h / 7d` | card velocity | Counts the card's own transactions in a **past** window `[t-w, t)`. Honest twin of the M2 leak `card_txn_count_next_24h`. |
| `card_amount_sum_prev_24h` | card velocity | Sum of the card's prior-24h spend (prefix-sum + searchsorted). |
| `secs_since_prev_txn` | card velocity | Gap to the immediately previous transaction on the card. |
| `amount_vs_card_prior_mean` | card velocity | This amount over the card's expanding mean of **prior** amounts. |
| `merchant_txns_seen` | merchant | Count of the merchant's transactions strictly before `t`. |
| `merchant_fraud_rate_asof` | merchant | Chargebacks on this merchant **already reported** by `t`, over transactions seen by `t`, smoothed by a Beta(1, 650) prior. Honest twin of the M2 leak `merchant_fraud_rate_lifetime`. |

## Honest offline result

A linear probe on a strict past → future split (a sanity check, **not** the final
model — that is Milestone 4):

| Feature set | Temporal-split AUC |
|---|---|
| Robust features (static + velocity) | **≈ 0.75** |
| Diagnostic merchant as-of features alone | **≈ 0.14** (inverted — see finding) |

Compare to Milestone 2's leaky ≈ 1.0. The honest number is lower and *real*.

## The label-delay finding (why 0.14, and why it's not a bug)

`merchant_fraud_rate_asof` is point-in-time-correct but **inverted on the drift
window**, and that is the single most important lesson of this milestone:

- A chargeback arrives 30–60 days after the fraud.
- The ring's merchants switch on at month 14 and are attacked heavily.
- At the moment of each ring transaction, **that merchant's own fraud has not been
  reported yet**, so its as-of risk is still near the prior — it looks *clean*.
- Meanwhile long-lived honest merchants have accumulated a small, *reported*
  baseline rate, so they look slightly *riskier* by comparison.

Net effect: honest merchant-reputation is highest for the safe merchants and lowest
for the ones being actively attacked. The feature points the wrong way — because the
label delay **blinds you to new attacks precisely when they begin.**

The fix is not a cleverer historical-rate feature. It is:
- a **tree model** (M4), which handles the non-monotonic signal without being misled
  by the sign;
- **drift monitoring** (M8), which watches the input/score distribution rather than
  waiting for labels;
- **retraining** (M9), which refreshes the model once the delayed labels arrive.

This is why fraud systems cannot rely on merchant reputation alone, and it is the
narrative thread that ties the back half of this project together.
