# Online Scoring Service (Milestone 6)

A FastAPI service that scores one transaction and returns a decision with reason
codes in a few milliseconds.

## Run it

```powershell
$env:LS_API_KEY = "choose-a-secret"     # required; no key, no scoring
$env:PYTHONPATH = "src"
uvicorn lonestar.serving:app --port 8000
```

The app loads `models/fraud_model.joblib`, `models/model_card.json`, and
`models/decision_policy.json` on the first request (point it elsewhere with
`LS_MODEL_DIR`).

## Endpoints

### `GET /health`
No auth. Returns `{"status": "ok"}` — for load balancers and readiness checks.

### `POST /score`
Requires `Authorization: Bearer <LS_API_KEY>`. Body:

```json
{
  "transaction_id": "abc-123",
  "features": { "log_amount": 3.9, "hour": 14, "...": "all 20 model features": 0 }
}
```

Response:

```json
{
  "transaction_id": "abc-123",
  "fraud_probability": 0.0535,
  "decision": "APPROVE",
  "threshold": 0.5804,
  "segment": "CHIP",
  "reason_codes": [
    { "feature": "log_amount", "contribution": -1.2039, "direction": "lowers" }
  ],
  "model_version": "2026-07-22T02:22:00+00:00",
  "latency_ms": 3.3
}
```

## What the service guarantees

- **Auth on every score.** Missing or wrong bearer token → `401`. If `LS_API_KEY`
  is unset the service refuses to score (`503`) rather than run open.
- **Train/serve schema safety.** The request's feature names are checked against
  the exact `feature_columns` in the model card. A missing or unknown feature is a
  `422` with the offending names — never a silently misaligned vector.
- **The decision uses the M5 policy.** The per-segment threshold is looked up from
  `decision_policy.json` (segment recovered from the entry-mode one-hots), so the
  API and the offline economics agree by construction.
- **Reason codes come from exact TreeSHAP.** XGBoost's native `pred_contribs`
  gives per-feature contributions in ~3 ms — the top five, signed, ship with every
  decision. No extra `shap` dependency.
- **Latency.** Single-row predict + TreeSHAP is comfortably under the 100 ms p99
  budget (`test_latency_p99_under_budget`).

## Where the features come from

This service scores a feature vector that is handed to it. Computing the stateful
features (card velocity, as-of merchant risk) from raw events is the feature
pipeline's job (Milestone 3), and wiring a low-latency feature store with
**train/serve parity** is Milestone 7. Keeping scoring and feature computation as
separate concerns is deliberate: the scorer stays fast, stateless, and easy to
test.
