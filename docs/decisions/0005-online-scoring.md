# ADR 0005 — Online scoring service design

## Context

The model and decision policy exist as artifacts; production needs to score a
single live transaction quickly, safely, and explainably. Key questions: how to
explain each decision, how to keep the serving feature vector aligned with
training, how to authenticate, and where feature computation lives.

## Decisions

1. **Reason codes via XGBoost's native `pred_contribs`, not the `shap` library.**
   XGBoost computes exact TreeSHAP contributions internally in ~3 ms for one row.
   This avoids a heavy extra dependency and its version friction, and returns the
   same values a reviewer would expect from TreeSHAP.

2. **Feature-schema validation against the model card.** The request's feature
   names are checked against the exact `feature_columns` recorded at training
   time. A mismatch is a `422`, never a silently reordered or zero-filled vector —
   the most common and most dangerous serving bug.

3. **The decision uses the M5 policy, not a hardcoded 0.5.** The per-segment
   threshold is loaded from `decision_policy.json`, so the live API and the
   offline economics memo can never disagree about what "decline" means.

4. **Bearer-token auth read from `LS_API_KEY`.** Secrets come from the
   environment; the service refuses to score (`503`) if no key is configured,
   rather than defaulting to open.

5. **Lazy artifact loading behind an app factory.** `create_app(model_dir)` builds
   an app without touching disk at import time; artifacts load on the first
   request. This keeps imports cheap and lets tests point the app at a temp model
   directory.

6. **Scoring and feature computation are separate concerns.** The service scores a
   feature vector handed to it. Stateful feature computation (velocity, as-of
   risk) belongs to the pipeline (M3), and a low-latency feature store with
   train/serve parity is M7. This keeps the scorer fast, stateless, and trivially
   testable.

## Consequences

- Every decision is explainable and auditable, with signed top-feature reasons.
- A drifted or malformed request fails loud and early instead of scoring garbage.
- The API's notion of "decline" is guaranteed identical to the economics memo's.
- The p99 latency budget (100 ms) is met with wide margin.
- The open question of serving the stateful features with exact parity is
  explicitly deferred to M7, where it is the headline.
