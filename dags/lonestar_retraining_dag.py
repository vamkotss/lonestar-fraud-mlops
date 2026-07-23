"""Airflow DAG: monthly drift check -> conditional retraining -> guarded promotion.

This orchestrates the pieces built in Milestones 8 and 9. The important design
choice is that **all the logic lives in ``lonestar``**, not in this file. The DAG
is a thin scheduler wrapper: every task calls a function that is unit-tested on
its own and runnable from the CLI without Airflow. That means the pipeline can be
tested, debugged, and re-run locally, and Airflow is not a dependency of CI.

Flow::

    check_drift ──> (drift?) ──yes──> train_challenger ──> evaluate ──> decide
                        │                                                 │
                        no                                       promote / hold
                        │                                                 │
                        └──────────────> skip ─────────────────────────────

Schedule: monthly. Retraining more often than labels mature is pointless -- see
``LABEL_MATURITY_DAYS`` in ``lonestar.retraining``.

To run this for real, install Airflow separately and point ``AIRFLOW_HOME`` at a
directory containing this ``dags/`` folder. Airflow is deliberately NOT in
``requirements.txt``: it is a heavy dependency that would slow CI without testing
anything the unit tests do not already cover.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Airflow is optional: importing this module without it should not explode, so the
# file can still be parsed and structurally checked in CI.
try:
    from airflow import DAG
    from airflow.operators.python import BranchPythonOperator, PythonOperator

    AIRFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when Airflow is absent
    DAG = None
    PythonOperator = BranchPythonOperator = None
    AIRFLOW_AVAILABLE = False

DATA_DIR = Path(os.environ.get("LS_DATA_DIR", "data/raw"))
MODELS_DIR = Path(os.environ.get("LS_MODEL_DIR", "models"))
REPORTS_DIR = Path(os.environ.get("LS_REPORTS_DIR", "reports"))

DEFAULT_ARGS = {
    "owner": "lonestar-risk",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}


# --------------------------------------------------------------------------- #
# Task callables -- thin wrappers over tested library code
# --------------------------------------------------------------------------- #
def task_check_drift(**context) -> dict:
    """Run the M8 monitor and push the result to XCom."""
    import joblib
    import pandas as pd

    from lonestar.decisions import _score_frame
    from lonestar.monitoring import monitor

    df = _score_frame(
        Path("data/features/features.parquet"), DATA_DIR, MODELS_DIR / "fraud_model.joblib"
    )
    report = monitor(df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2))
    # Keep only what the branch needs; XCom is not a data lake.
    del joblib, pd
    return {
        "first_alert_month": report["first_alert_month"],
        "n_alerting_windows": sum(1 for w in report["windows"] if w["status"] == "ALERT"),
    }


def task_should_retrain(**context) -> str:
    """Branch: retrain only when the monitor actually alerted."""
    ti = context["ti"]
    drift = ti.xcom_pull(task_ids="check_drift") or {}
    return "run_retraining_cycle" if drift.get("first_alert_month") is not None else "skip_retraining"


def task_run_retraining_cycle(**context) -> dict:
    """Train a challenger, compare against the champion, apply the guardrails."""
    import joblib
    import pandas as pd

    from lonestar.retraining import run_cycle

    tx = pd.read_parquet(DATA_DIR / "transactions.parquet")
    labels = pd.read_parquet(DATA_DIR / "chargeback_labels.parquet")
    champion = joblib.load(MODELS_DIR / "fraud_model.joblib")

    report = run_cycle(tx, labels, champion)
    challenger = report.pop("challenger")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "retraining_cycle.json").write_text(json.dumps(report, indent=2))

    if report["decision"]["promote"]:
        # Keep the outgoing champion so a promotion can be rolled back.
        joblib.dump(champion, MODELS_DIR / "fraud_model.previous.joblib")
        joblib.dump(challenger, MODELS_DIR / "fraud_model.joblib")

    return report["decision"]


def task_skip_retraining(**context) -> str:
    return "no drift alert; champion retained"


# --------------------------------------------------------------------------- #
# DAG definition
# --------------------------------------------------------------------------- #
if AIRFLOW_AVAILABLE:  # pragma: no cover - requires Airflow installed
    with DAG(
        dag_id="lonestar_fraud_retraining",
        description="Monthly drift check, conditional challenger training, guarded promotion.",
        default_args=DEFAULT_ARGS,
        start_date=datetime(2024, 1, 1),
        schedule="@monthly",
        catchup=False,
        tags=["fraud", "mlops", "retraining"],
    ) as dag:
        check_drift = PythonOperator(
            task_id="check_drift", python_callable=task_check_drift
        )
        should_retrain = BranchPythonOperator(
            task_id="should_retrain", python_callable=task_should_retrain
        )
        run_retraining_cycle = PythonOperator(
            task_id="run_retraining_cycle", python_callable=task_run_retraining_cycle
        )
        skip_retraining = PythonOperator(
            task_id="skip_retraining", python_callable=task_skip_retraining
        )

        check_drift >> should_retrain >> [run_retraining_cycle, skip_retraining]
