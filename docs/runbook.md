# Runbook — Operating the Fraud Pipeline

How to start, stop, and recover the system. Filled in as operable components land;
for now it covers the local stack.

## Start the local stack
```
docker compose up -d          # Postgres (and later MLflow, Airflow)
```

## Stop
```
docker compose down           # add -v to also remove volumes (wipes data)
```

## Regenerate the dataset
```
python -m lonestar.generate --mode history      # (Milestone 1)
```

## Common failures
- **Port 5434 already in use** — another Postgres is bound. Find and stop it, or
  change `LS_DB_PORT`. (P2 uses 5433; this project uses 5434 to avoid collision.)
- **Container won't start / disk full** — check free space before generating data;
  this project produces several million rows.

*Backfill, retraining, and rollback procedures are added with Milestones 3, 9, and 10.*
