# ADR 0008 — Challenger/champion retraining under delayed labels

## Context

Milestone 8 detects a regime change the month it happens. Something then has to
respond: retrain the model. Two facts constrain how.

First, **labels are 30-60 days late**. The transactions from the last two months
look almost fraud-free simply because their chargebacks have not been filed yet.
Training on them would teach the model that an active attack is legitimate — the
delayed-label version of the leakage problem Milestone 2 was built around.

Second, **automatic promotion is dangerous**. A pipeline that ships whatever model
scored highest this month will eventually ship a worse one, because a single
metric on a small window is noisy and an average can improve while a whole channel
collapses.

## Decisions

1. **Gate training on label maturity.** Training data is cut at
   `as_of - LABEL_MATURITY_DAYS` (60, matching the generator's maximum delay). The
   most recent *matured* slice (`EVAL_WINDOW_DAYS`, 60) is held out to compare
   candidates. Everything newer is treated as unlabelled and excluded.

   Consequence, stated plainly: a ring starting in month 13 cannot be *learned*
   until roughly month 15. Monitoring gives early warning; it does not remove the
   wait. Covering that gap is an operational matter (tighten thresholds, route more
   volume to review) and belongs in the runbook, not in this pipeline.

2. **Labels are censored, not just filtered.** Candidates are trained on
   `build_training_frame(..., as_of=...)`, so only chargebacks actually *reported*
   by the run date are visible — the honest view on the day the job runs, and the
   same point-in-time discipline as ADR 0003.

3. **Three promotion guardrails, all of which must pass:**
   - at least `MIN_EVAL_POSITIVES` (20) fraud cases in the evaluation window;
   - at least `MIN_RELATIVE_IMPROVEMENT` (+5%) relative PR-AUC over the champion;
   - no segment losing more than `MAX_SEGMENT_REGRESSION` (0.05) absolute PR-AUC.

   A failed guardrail is a **HOLD**, recorded with reasons. The champion stays.

4. **Logic lives in the library; Airflow is a thin wrapper.** Every task in
   `dags/lonestar_retraining_dag.py` calls a unit-tested function that also runs
   from the CLI. Airflow is deliberately **not** in `requirements.txt` — it is a
   heavy dependency that would slow CI without covering anything the unit tests
   miss. The DAG is validated structurally (it parses, defines its task callables,
   and imports without Airflow present).

5. **Promotion is reversible.** The DAG writes the outgoing champion to
   `fraud_model.previous.joblib` before overwriting, so a bad promotion can be
   rolled back without retraining.

## Consequences

- Demonstrated both ways on the CI dataset: a champion trained *before* the ring
  scores **PR-AUC 0.003** on the ring-heavy evaluation window (below chance), while
  a challenger trained on matured post-ring labels reaches **0.515** → PROMOTE.
  Re-running against an already-current champion → HOLD. The pipeline earns its
  promotions rather than rubber-stamping them.
- `--champion-as-of` simulates a stale champion so the promote path is
  reproducible from the command line for demos and documentation.
- Retraining more often than labels mature is pointless, so the DAG is monthly.
- The honest limitation: this cycle can only react to attacks the labels have
  caught up with. Faster response requires operational levers, not a better model.
