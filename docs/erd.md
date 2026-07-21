# Entity-Relationship Diagram — Fraud Warehouse

This is the schema **designed at Milestone 1**, before any data exists. Designing
the schema is itself the skill; the tables below anticipate what every later
milestone needs, so the structure doesn't get rebuilt mid-project.

The single most important design choice is that **labels are separated from
transactions**. A fraud label (chargeback) arrives 30–60 days *after* the
transaction. Modeling that as a nullable column on `transactions` would quietly
invite leakage — you'd be tempted to train on labels that, on the transaction
date, did not yet exist. A separate `chargeback_labels` table with its own
`reported_at` timestamp forces every feature and every training split to respect
the "what did we know, and when did we know it" boundary.

```mermaid
erDiagram
    CARDS ||--o{ TRANSACTIONS : "makes"
    TRANSACTIONS ||--o| CHARGEBACK_LABELS : "may be disputed as"
    TRANSACTIONS ||--o{ PREDICTIONS : "scored by"
    MODELS ||--o{ PREDICTIONS : "produces"
    TRANSACTIONS ||--o{ FEATURES : "described by"

    CARDS {
        text card_id PK
        text customer_segment "retail | affluent | small_business"
        date issued_on
        text home_state
    }

    TRANSACTIONS {
        text transaction_id PK
        text card_id FK
        timestamptz authorized_at "event time (timezone-corrected)"
        numeric amount
        text merchant_name "raw, un-normalized (AMZN Mktp US*2K4)"
        text merchant_category_code "4% missing"
        text channel "card_present | ecommerce | recurring"
        text auth_or_capture "duplicate auth/capture pairs land here"
    }

    CHARGEBACK_LABELS {
        text transaction_id PK "FK to transactions"
        boolean is_fraud
        timestamptz reported_at "30-60 days AFTER authorized_at - the moving horizon"
    }

    FEATURES {
        text transaction_id PK "FK to transactions"
        timestamptz as_of "point-in-time the features were computed for"
        numeric amount_zscore_7d
        int txn_count_1h
        numeric merchant_risk_score
        boolean is_new_merchant_for_card
    }

    MODELS {
        text model_id PK
        text algorithm "logistic_regression | xgboost"
        timestamptz trained_at
        text training_window "temporal split boundaries"
        text stage "staging | production | archived"
        numeric pr_auc
    }

    PREDICTIONS {
        text prediction_id PK
        text transaction_id FK
        text model_id FK
        numeric fraud_score
        boolean decision "approve | decline"
        text top_reasons "SHAP top-3 reason codes (M6)"
        timestamptz scored_at
        text path "online | batch (parity-tested in M7)"
    }
```

## Why each table exists

- **cards** — the customer dimension; `customer_segment` drives per-segment
  thresholds in the decision layer (M5), because the cost of a false decline
  differs by segment.
- **transactions** — the raw event stream, landed faithfully with its defects
  (missing MCCs, chaotic merchant names, duplicate auth/capture pairs) so cleaning
  is a documented, tested step rather than a silent one.
- **chargeback_labels** — separated precisely because labels are late; `reported_at`
  is what makes point-in-time-correct training possible.
- **features** — computed as-of a point in time; `as_of` is the guardrail against
  using the future to predict the past.
- **models** — the registry metadata mirror of what MLflow tracks; `stage` carries
  the champion/challenger promotion state (M9).
- **predictions** — every score, from either serving path, with reason codes; `path`
  lets the M7 parity test prove online and batch agree.
