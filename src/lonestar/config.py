"""Central configuration, read from environment variables.

The Repository Standard forbids hardcoded paths or credentials anywhere in src/.
Every setting is read here, once, from the environment (loaded from .env in local
development via python-dotenv), so the same code runs locally, in CI, and in a
container by changing env vars rather than editing code.
"""

from __future__ import annotations

import os


def _get(key: str, default: str | None = None) -> str:
    value = os.environ.get(key, default)
    if value is None:
        raise KeyError(f"required environment variable {key!r} is not set")
    return value


# Database connection (Postgres in Docker; port 5434 to avoid colliding with P2's 5433).
DB_HOST = os.environ.get("LS_DB_HOST", "localhost")
DB_PORT = os.environ.get("LS_DB_PORT", "5434")
DB_USER = os.environ.get("LS_DB_USER", "lonestar")
DB_PASSWORD = os.environ.get("LS_DB_PASSWORD", "lonestar")
DB_NAME = os.environ.get("LS_DB_NAME", "fraud")

# Fixed seed so every generated dataset is reproducible.
SEED = int(os.environ.get("LS_SEED", "20260721"))


def database_url() -> str:
    """SQLAlchemy connection string for the fraud warehouse."""
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
