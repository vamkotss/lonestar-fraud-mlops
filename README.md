# Lonestar Card Fraud Detection — Full MLOps

[![CI](https://github.com/vamkotss/lonestar-fraud-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/vamkotss/lonestar-fraud-mlops/actions/workflows/ci.yml)

**A regional card issuer loses ~$0.9M a year to fraud — but loses *more* to false declines that block good customers at checkout and send them to competitors. Its rules engine has 200+ thresholds nobody dares touch. This project replaces it with an ML system that minimizes *total dollar loss*, not a classification score: leakage-audited features, models validated on time (never random splits), a scorer that runs live and in batch and provably agrees with itself, drift monitoring that catches a new fraud ring, and retraining that promotes a challenger only through a gate.**

> The headline isn't the model's accuracy. Accuracy is meaningless when 0.15% of transactions are fraud. The headline is that every decision is **explained, costed in dollars, temporally honest, and monitored** — the things that make an ML system trustworthy in production.

---

## Why this is hard (and why that's the point)

Fraud is one of the most instructive ML problems because the naive approach fails in instructive ways:

- **Extreme imbalance** (~0.15% fraud) — accuracy is useless; a model that predicts "never fraud" is 99.85% accurate and worthless.
- **Labels arrive 30–60 days late** — you can't random-split the data; the label horizon moves, and a random split lets the future train the past.
- **Leakage is everywhere** — three traps are planted in the data, and Milestone 2 is an audit that must find all three.
- **Two costs pull opposite ways** — catching more fraud means more false declines, and false declines cost more. The decision layer prices this tradeoff in dollars.

---

## Stack

`Python` · `PostgreSQL` · `XGBoost` · `scikit-learn` · `MLflow` · `SHAP` · `Evidently` · `FastAPI` · `Airflow` · `Docker` · `GitHub Actions`

Each choice is defended in an ADR under [`docs/decisions/`](docs/decisions/).

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and walkthrough. In brief:

```
generate → land (Postgres) → point-in-time features → train (temporal splits, MLflow)
   → decide (cost matrix) → serve (online API + batch, parity-tested)
   → monitor drift → retrain (challenger/champion gate)
```

---

## Status

Built milestone by milestone (see [`CHANGELOG.md`](CHANGELOG.md)):

- [x] **M1** — Scaffold, business brief, ERD, green CI
- [ ] M2 — Transaction generator + leakage audit
- [ ] M3 — Point-in-time feature pipeline
- [ ] M4 — Models + MLflow tracking
- [ ] M5 — Expected-loss thresholding + economics memo
- [ ] M6 — Online scoring API + SHAP reason codes
- [ ] M7 — Batch scorer + parity test
- [ ] M8 — Drift monitoring
- [ ] M9 — Automated retraining loop
- [ ] M10 — Model card + runbook

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q          # scaffold smoke tests pass
ruff check .       # clean
```

The Postgres stack and data generation arrive with Milestone 2.

---

## Docs

- [Business brief](docs/business_brief.md) — the problem, stakeholders, KPIs, value
- [Architecture](docs/architecture.md) · [ERD](docs/erd.md) · [Data dictionary](docs/data_dictionary.md)
- [Decisions (ADRs)](docs/decisions/) · [Runbook](docs/runbook.md)

---

## Author

**Sri Vamsi Kota** — MS Business Analytics & AI, UT Dallas · [github.com/vamkotss](https://github.com/vamkotss)
