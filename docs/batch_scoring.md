# Batch Scoring & Train/Serve Parity (Milestone 7)

The batch scorer decisions a whole table of transactions at once — for nightly
runs, backfills, or what-if analysis. It is the offline twin of the online service
(Milestone 6), and it is built so the two can never disagree.

## Run it

```powershell
$env:PYTHONPATH = "src"
python -m lonestar.batch `
  --features data\features\features.parquet `
  --model    models\fraud_model.joblib `
  --policy   models\decision_policy.json `
  --out      data\scored\scored.parquet
```

Output columns: `transaction_id`, `fraud_probability`, `segment`, `threshold`,
`decision`.

## The parity guarantee

Serving skew — the online model scoring a transaction differently from the offline
job — is one of the most common and most damaging failures in production ML. This
project rules it out two ways:

**Decision parity (online == batch).** Both paths call the *same* shared core,
`lonestar.scoring`, for segment recovery, threshold lookup, and the decline rule,
over the *same* model and policy artifacts. They cannot drift because there is only
one implementation. `test_online_batch_parity` proves it empirically: it scores the
same 40 transactions through the live API and through the batch job and asserts the
probabilities match to within rounding and every decision is identical.

**Feature parity (train == serve).** Both training and scoring build features from
the single definition in `lonestar.features` (`FEATURE_COLUMNS`). There is no
separate "serving feature code" to fall out of sync — the columns, order, and
semantics are shared by construction, and the online service additionally validates
each request against the exact `feature_columns` recorded in the model card.

## Why a batch scorer at all

- **Backfills / reprocessing** when the model or policy changes.
- **What-if analysis**: re-score history under a new cost model to estimate impact
  before shipping a threshold change.
- **Throughput**: vectorised `predict_proba` over millions of rows is far faster
  than one HTTP call each.

The online service handles the real-time single-transaction path; the batch scorer
handles everything bulk — and they agree, by design and by test.
