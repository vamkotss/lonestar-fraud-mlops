# Data Dictionary

Every table and column: type, meaning, source, and quality rules. Populated as
each table is built; the planned schema is in `erd.md`.

## raw.transactions  *(built in Milestone 1)*

| Column | Type | Meaning | Quality rule |
|---|---|---|---|
| transaction_id | text | Unique transaction identifier | Primary key, non-null |
| card_id | text | The card that made it | FK to cards |
| authorized_at | timestamptz | Event time, timezone-corrected | Source is timezone-naive; corrected on load |
| amount | numeric | Transaction amount (USD) | Non-null |
| merchant_name | text | Raw merchant string | Deliberately un-normalized (e.g. `AMZN Mktp US*2K4`) |
| merchant_category_code | text | MCC | ~4% missing (real-world gap) |
| channel | text | card_present / ecommerce / recurring | Enumerated |
| auth_or_capture | text | auth vs capture | Duplicate auth/capture pairs occur |

## raw.chargeback_labels  *(built in Milestone 1)*

| Column | Type | Meaning | Quality rule |
|---|---|---|---|
| transaction_id | text | The disputed transaction | FK to transactions |
| is_fraud | boolean | Confirmed fraud or not | Non-null |
| reported_at | timestamptz | When the label arrived (30–60 days later) | Always > the transaction's authorized_at |

*Feature, model, and prediction tables are documented as Milestones 3, 4, and 6 build them.*
