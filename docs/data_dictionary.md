# Data Dictionary — Generated Data (Milestone 1)

Two tables are produced by `src/lonestar/generation.py`. Fraud is **never** a
column on `transactions` — it lives only in `chargeback_labels`, and it arrives
30–60 days late (see ADR 0001).

## `transactions`

The raw event stream, landed with its defects intact so cleaning is a tested,
documented step later — not a silent one.

| Column | Type | Notes |
|---|---|---|
| `transaction_id` | string | Unique per **row**. Auth and capture are separate rows. |
| `auth_code` | string | Logical **purchase** key. Links an auth to its capture. |
| `card_id` | string | Cardholder card. |
| `account_id` | string | Owning account (a few cards per account). |
| `merchant_id` | string | **Clean** canonical merchant id — the entity-resolution answer key. |
| `merchant_name` | string | **Messy** descriptor. One `merchant_id` → many of these. |
| `mcc` | Int64 (nullable) | Merchant category code. **~4% NULL** by design. |
| `amount` | float | Transaction amount, USD. Captures may differ slightly (tips/partials). |
| `currency` | string | `USD`. |
| `entry_mode` | string | `CHIP` / `CONTACTLESS` / `SWIPE` / `ECOM` / `MANUAL`. |
| `pos_country` | string | Usually `US`; the ring skews foreign. |
| `event_ts` | datetime (**tz-naive**) | Event time (auth time on auth rows, capture time on capture rows). |
| `response_code` | string | `APPROVED`. |
| `event_type` | string | `AUTH` or `CAPTURE`. |
| `dispute_reason_code` | string | ⚠️ **LEAK A** — see `leakage_traps.md`. |
| `merchant_fraud_rate_lifetime` | float | ⚠️ **LEAK B** — see `leakage_traps.md`. |
| `card_txn_count_next_24h` | int | ⚠️ **LEAK C** — see `leakage_traps.md`. |

## `chargeback_labels`

One row **per fraudulent purchase**. Non-fraud purchases have no row (their label
is the *absence* of a chargeback after the maturity window). `reported_at` is what
makes point-in-time-correct training possible.

| Column | Type | Notes |
|---|---|---|
| `auth_code` | string | FK to the fraudulent purchase. |
| `transaction_id` | string | The purchase's AUTH row, for convenience joins. |
| `is_fraud` | bool | Always `True` (a chargeback table is fraud-only). |
| `fraud_type` | string | `BASELINE` or `RING` (month-14 ring). |
| `chargeback_amount` | float | Disputed amount. |
| `reason_code` | string | Visa/MC-style dispute code. |
| `event_ts` | datetime (**tz-naive**) | Original transaction time. |
| `reported_at` | datetime (**tz-naive**) | When the chargeback landed = `event_ts` + 30–60 days. |
