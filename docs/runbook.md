# Runbook — Fraud Model Operations

Who this is for: whoever is on call when the drift monitor pages, or whoever
inherits this system. It assumes no prior context beyond this repository.

---

## 1. The single most important thing to understand

**Chargebacks arrive 30–60 days after the transaction.**

Everything awkward about operating this system follows from that one fact:

- You cannot tell how much fraud happened last week. Those chargebacks have not
  been filed yet.
- The drift monitor can tell you an attack started **today**, because it watches
  inputs and scores rather than outcomes.
- But you cannot **retrain** on today's attack until its labels mature (~60 days).

So there is a real window — roughly two months — where you *know* you are being
attacked and cannot yet fix the model. That window is covered by operational
levers, not by modelling. Section 3 is that playbook.

---

## 2. Drift alert fired. What now?

The monitor (`python -m lonestar.monitoring`) raises **WARN** or **ALERT** per
month, per channel. An ALERT means a channel's rate of high-scoring transactions
is both ≥2× its baseline and ≥3σ outside normal variation.

### Step 1 — Confirm it is real, not an artifact

```bash
python -m lonestar.monitoring --features data/features/features.parquet \
    --raw data/raw --model models/fraud_model.joblib --out reports
```

Check `docs/drift_monitoring.md` and ask:

- **Which channel?** A single channel spiking is the signature of a targeted
  attack. All channels moving together is more often an upstream data problem.
- **Is the window complete?** A partial month produces junk drift on calendar
  features. Windows under 2,000 rows are skipped for exactly this reason — if you
  changed that, change it back.
- **Did anything upstream change?** A new merchant onboarding, a payment-processor
  migration, or a schema change can all shift feature distributions without any
  fraud involved. Rule this out before assuming an attack.

### Step 2 — Look at what actually changed

`reports/drift_report.json` lists the top drifting features per window. If the
drift is concentrated in entry mode, geography, and amount band, that is a fraud
pattern. If it is in `mcc_missing` or timestamp features, suspect a pipeline bug.

### Step 3 — Decide whether to act now or wait

| Situation | Action |
|---|---|
| One channel, fraud-shaped features, sustained ≥2 windows | Treat as an attack → Section 3 |
| One channel, single window, marginal σ | Watch one more cycle |
| All channels shifted at once | Suspect upstream data → page the data team |
| Drift on data-quality features only | Pipeline bug, not fraud |

---

## 3. Confirmed attack, labels not yet mature

This is the two-month gap. You cannot retrain yet. You *can*:

### Lever 1 — Tighten the threshold for the affected channel

`models/decision_policy.json` is a small file, deliberately decoupled from the
model binary so it can be changed without retraining. Lowering the threshold on
the attacked channel declines more transactions there.

**This is a cost decision, not a technical one.** The economics memo
(`docs/economics_memo.md`) quantifies the trade: at the current policy, a false
decline costs a fixed friction charge plus lost margin, while a missed fraud costs
the whole amount. During an active ring the effective fraud rate in that channel
is elevated, which justifies temporarily accepting more false declines.

Get sign-off from whoever owns the false-decline budget. Record the change and the
date — a threshold changed in an incident and never reverted is a classic source
of quiet, long-term customer harm.

### Lever 2 — Route the uncertain band to human review

The current policy is binary (approve/decline). The natural extension — approve /
**review** / decline — routes a middle band to analysts. If a review queue exists,
widening it is lower-risk than declining outright, because a human sees the
transaction before a customer is turned away.

### Lever 3 — Escalate to the fraud strategy team

Model thresholds are one control among many. Merchant-level blocks, velocity
rules, and step-up authentication all act faster than a model change and are not
subject to the label delay.

### What NOT to do

- **Do not retrain on recent data to "catch up".** Transactions from the last 60
  days are mostly labelled fraud-free because their chargebacks have not been
  filed. Training on them teaches the model that the attack is legitimate. The
  maturity gate in `lonestar.retraining` exists to make this mistake impossible;
  do not work around it.
- **Do not disable the monitor because it is noisy.** If it is firing too often,
  tune the control limits deliberately and write down why.

---

## 4. Labels have matured. Retrain.

```bash
# See what a challenger would do, without touching the champion:
python -m lonestar.retraining --data data/raw --models models

# If the verdict is PROMOTE and you accept it:
python -m lonestar.retraining --data data/raw --models models --promote
```

The cycle trains a challenger on matured labels, compares it to the champion on a
held-out matured window, and promotes **only** if all three guardrails pass:

1. ≥20 fraud cases in the evaluation window (enough to decide on)
2. ≥+5% relative PR-AUC over the champion (a material win, not noise)
3. No channel losing >0.05 absolute PR-AUC (no segment sacrificed for the average)

A **HOLD** is a normal, healthy outcome. It means the champion is still the best
model available. Read the reasons in `docs/retraining.md` before overriding.

After promoting, **re-tune the decision policy** — thresholds calibrated for the
old model do not transfer to a new one:

```bash
python -m lonestar.decisions --features data/features/features.parquet \
    --raw data/raw --model models/fraud_model.joblib --out models
```

Then revert any emergency threshold changes from Section 3.

---

## 5. Rollback

The retraining DAG saves the outgoing champion before overwriting:

```bash
cp models/fraud_model.previous.joblib models/fraud_model.joblib
python -m lonestar.decisions --features data/features/features.parquet \
    --raw data/raw --model models/fraud_model.joblib --out models
```

Re-tuning the policy after a rollback is not optional, for the same reason as
above. Then restart the scoring service so it reloads the artifacts.

**Roll back if**, after a promotion: the decline rate moves sharply in any channel,
the service's latency budget is breached, or the score distribution looks nothing
like the previous model's. You do not need to wait for chargebacks to justify a
rollback — that is 60 days you do not have.

---

## 6. Scoring service issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `503 scoring auth is not configured` | `LS_API_KEY` unset | Set it; the service refuses to run open by design |
| `422` with `missing_features` | Caller's feature set drifted from the model card | Fix the caller, or retrain and redeploy together — do **not** pad missing features with zeros |
| `503 model artifact missing` | `LS_MODEL_DIR` wrong, or artifacts not deployed | Check the path; artifacts load lazily on first request |
| Latency above budget | Cold start, or an oversized model | First request loads artifacts; pre-warm with `/health` after deploy |

The online and batch paths share one decision core (`lonestar.scoring`), so they
cannot disagree — `test_online_batch_parity` enforces it. If you ever see the API
and a batch run give different decisions for the same transaction, that is a
serious bug, not a rounding difference.

---

## 7. Routine checks

**Monthly** (or whenever the DAG runs):

```bash
python -m lonestar.monitoring --features data/features/features.parquet \
    --raw data/raw --model models/fraud_model.joblib --out reports
python -m lonestar.retraining --data data/raw --models models
python -m lonestar.governance          # regenerate the model card
```

**Before any deploy**: `ruff check src tests dags` and `pytest -q` must both be
clean, and `python -m lonestar.governance --check` must report every governance
artifact present.

---

## 8. Escalation

| Situation | Who |
|---|---|
| Sustained ALERT across multiple channels | Fraud strategy + data engineering |
| Threshold change beyond the agreed false-decline budget | Whoever owns that budget |
| Suspected upstream data corruption | Data engineering, before touching the model |
| Model behaving unexplainably in production | Roll back first (Section 5), diagnose second |

Rolling back is always available and always cheap. Diagnosing in production while
customers are declined is neither.
