# ADR 0002 — XGBoost over deep neural networks for the classifier

## Context

The task is tabular fraud classification on a few million rows with extreme class
imbalance (~0.15% fraud). We need a model that is accurate, cheap to train and
serve (<100ms scoring), and interpretable enough to explain a decline to ops.

## Options

1. **Logistic regression** — the honest baseline; interpretable, fast, but limited
   on non-linear feature interactions.
2. **XGBoost (gradient-boosted trees)** — state-of-the-art on tabular data at this
   scale, handles imbalance and mixed feature types well, fast to serve, and
   explainable via SHAP.
3. **Deep neural network** — strong on images/text/sequences, but on modest-scale
   tabular data it rarely beats gradient boosting, costs more to train and serve,
   and is harder to explain.

## Decision

Logistic regression as the baseline we must beat honestly, then **XGBoost** as the
production model.

## Consequences

- We get SHAP reason codes (M6) almost for free — the ops-facing explainability the
  model risk officer and the fraud analysts both need.
- Serving stays cheap and fast, keeping the API within its p99 budget.
- The ADR names the condition under which a neural approach *would* win (much larger
  scale, or raw-sequence features), which is the nuance an interviewer probes.
