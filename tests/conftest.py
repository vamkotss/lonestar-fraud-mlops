"""Shared fixtures. Generate ONE CI-scale dataset per test session.

We build the small-but-representative slice in memory (no file I/O) so every test
asserts against the same physics the full 5M-row run produces.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from lonestar.generation import GenConfig, generate

RING_ONSET = 14


@pytest.fixture(scope="session")
def ci_config() -> GenConfig:
    """A deterministic CI-scale config, independent of the ambient environment."""
    base = GenConfig.from_env()
    return GenConfig(**{**asdict(base), "n_purchases": 120_000, "seed": 20260721, "ci_mode": True})


@pytest.fixture(scope="session")
def dataset(ci_config):
    """(transactions, chargeback_labels, manifest) built once for the session."""
    return generate(ci_config)


@pytest.fixture(scope="session")
def tx(dataset) -> pd.DataFrame:
    return dataset[0]


@pytest.fixture(scope="session")
def labels(dataset) -> pd.DataFrame:
    return dataset[1]


@pytest.fixture(scope="session")
def manifest(dataset) -> dict:
    return dataset[2]


@pytest.fixture(scope="session")
def row_fraud(tx, labels) -> np.ndarray:
    """Per-ROW fraud flag, recovered the honest way: join to the label table.

    Fraud is never a column on ``tx`` (that's the whole ADR); a row is fraud iff
    its purchase (``auth_code``) has a chargeback.
    """
    fraud_auth = set(labels["auth_code"])
    return tx["auth_code"].isin(fraud_auth).to_numpy()
