# Lonestar Card Services — Card Fraud Detection with Full MLOps

![CI](https://github.com/vamkotss/lonestar-fraud-mlops/actions/workflows/ci.yml/badge.svg)

An end-to-end fraud detection system for a fictional card issuer: 5M transactions,
0.15% fraud, chargeback labels that arrive **30–60 days late**, and a fraud ring
that switches on midway through the timeline.

The interesting part is not the model. It is everything the delayed label breaks.

---

## Four findings this project is built around

**1. A leaky model scores 1.00 and is worthless.**
Three data leaks were deliberately planted in the source data, then hunted down
with a detector per leak archetype. On a strict temporal split, the leaky model
scores an AUC of **1.00**; the honest one scores **0.79**. That **+0.21 gap is the
cost of one careless join** — and a naive univariate screen catches only one of the
three leaks, which is why the audit uses structural detectors instead.
→ [`docs/leakage_audit_report.md`](docs/leakage_audit_report.md)

**2. The default 0.5 threshold would decline 16% of legitimate customers.**
Turning probabilities into decisions is an economics problem: a missed fraud costs
the full amount, a false decline costs a fixed friction charge plus lost margin.
Optimising **dollars** rather than error count, per channel, on a held-out
calibration window:

| Policy | Saved vs approve-all | Fraud $ caught | False declines |
|---|---|---|---|
| Default threshold 0.5 | $171,481 | 96.7% | **81,862 (16.3%)** |
| **Cost-optimal per-segment** | **$788,391** | 94.9% | **338 (0.067%)** |

4.6× the savings, **240× fewer false declines**, for the same model.
→ [`docs/economics_memo.md`](docs/economics_memo.md)

**3. Population drift metrics did not detect the fraud ring at all.**
The obvious monitoring design — PSI across all features and the score distribution
— was built first and **completely missed** a live attack. A ring touching ~1% of
cards is a rounding error in any average. Detection requires watching the **score
tail within each channel** against a control limit; that fires at **4–5σ in the
month the ring switches on**, using **no labels at all**.
→ [`docs/drift_monitoring.md`](docs/drift_monitoring.md)

**4. Monitoring buys early warning; it does not remove the wait.**
The monitor flags the attack on day one. But a *retrained* model still waits ~60
days for chargebacks to mature — training on immature labels would teach the model
that the live attack is legitimate. That gap is closed operationally, not
statistically.
→ [`docs/runbook.md`](docs/runbook.md)

---

## What's in here

| Stage | Module | What it does |
|---|---|---|
| Data | `generation.py` | Seeded 5M-row generator: imbalance, label delay, drift ring, 3 planted leaks, realistic mess |
| Audit | `audit.py` | Leak detection per archetype + honest-vs-leaky AUC |
| Features | `features.py` | Point-in-time features; a **future-proof test** proves none peek forward |
| Model | `modeling.py` | LR baseline vs XGBoost, temporal CV, MLflow tracking, PR-AUC selection |
| Decisions | `decisions.py` | Asymmetric cost model, per-segment cost-optimal thresholds |
| Serving | `serving.py` | FastAPI scorer: auth, schema validation, TreeSHAP reason codes, <100ms p99 |
| Batch | `batch.py` | Vectorised bulk scoring, sharing one decision core with the API |
| Monitoring | `monitoring.py` | Label-free segment tail drift detection with control limits |
| Retraining | `retraining.py` | Challenger/champion with label-maturity gating and promotion guardrails |
| Governance | `governance.py` | Auto-generated model card + a CI gate on documentation completeness |

Plus an Airflow DAG (`dags/`) orchestrating drift check → conditional retrain →
guarded promotion, and **9 architecture decision records** in
[`docs/decisions/`](docs/decisions/).

---

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH = "src"

python -m lonestar.generation --out data/raw                    # ~5M rows, ~45s
python -m lonestar.features   --data data/raw --out data/features
python -m lonestar.modeling   --features data/features/features.parquet --out models
python -m lonestar.decisions  --features data/features/features.parquet --raw data/raw \
                              --model models/fraud_model.joblib --out models
python -m lonestar.monitoring --features data/features/features.parquet --raw data/raw \
                              --model models/fraud_model.joblib --out reports
python -m lonestar.governance                                   # regenerate the model card
```

Run the scoring API:

```bash
export LS_API_KEY=your-secret
uvicorn lonestar.serving:app --port 8000
# interactive docs at http://127.0.0.1:8000/docs
```

Everything runs on a small representative dataset with `LS_CI_MODE=1` — same
imbalance, same delay, same ring, same leaks, seconds instead of minutes. That is
what CI uses.

---

## Engineering notes

**Temporal splits everywhere.** Labels arrive 30–60 days late, so a random split
leaks the future into training. Every split — model, thresholds, evaluation — is
by time. Thresholds are tuned on a *held-out calibration window*, because tuning
them on the model's own training data reads memorised in-sample scores that do not
transfer.

**Train/serve parity is structural, not aspirational.** The online API and the
batch job call one shared decision core over one set of artifacts, and
`test_online_batch_parity` scores the same transactions both ways and asserts
identical decisions.

**Two bugs only appeared at production scale.** In-sample threshold tuning and a
fixed monitoring cutoff were both invisible on 224K rows and broken on 5M — the
first switched the decision layer off entirely (0 declines out of 5,049,290), the
second left the drift monitor blind. Both are now documented in their ADRs with
regression tests. CI being green is not the same as being correct.

**Reason codes on every decision.** Exact TreeSHAP contributions via XGBoost's
native `pred_contribs` (~3ms), so an adverse decision can be explained rather than
asserted.

---

## Reading order

If you have five minutes: this README, then
[`docs/model_card.md`](docs/model_card.md).

If you have thirty: add [`docs/leakage_audit_report.md`](docs/leakage_audit_report.md),
[`docs/economics_memo.md`](docs/economics_memo.md), and
[`docs/drift_monitoring.md`](docs/drift_monitoring.md) — the three findings above,
with the working.

If you are going to operate it: [`docs/runbook.md`](docs/runbook.md).

---

*Synthetic data, generated for this project. The engineering is production-shaped;
the numbers are not a claim about any real portfolio.*
