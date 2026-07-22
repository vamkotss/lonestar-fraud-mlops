"""Turn the two-table (transactions + late labels) design into a modelling label.

Why this is its own tiny module:

    Fraud is NOT a column on ``transactions`` -- that is the whole point of the
    ERD (ADR 0001). A row is fraud only if its ``auth_code`` shows up in the
    ``chargeback_labels`` table, which arrives 30-60 days later. Every downstream
    step (EDA, the leakage audit, later the feature pipeline) needs the SAME
    definition of "is this row fraud", so it lives in one place.

    Note both the AUTH and the CAPTURE row of a fraudulent purchase share the
    same ``auth_code``, so membership correctly labels *both* rows of the pair.
"""

from __future__ import annotations

import pandas as pd


def attach_row_label(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """Return a boolean Series (aligned to ``transactions.index``): is this row fraud?

    Parameters
    ----------
    transactions
        The transactions table (must contain ``auth_code``).
    labels
        The ``chargeback_labels`` table (must contain ``auth_code`` and, if
        ``as_of`` is used, ``reported_at``).
    as_of
        If given, only labels whose chargeback had already been *reported* by this
        timestamp count as known fraud. This is how you avoid temporal leakage:
        on the day you score a transaction, you only know about chargebacks that
        have actually come back. If ``None`` (default), every eventual fraud is
        treated as known -- which is fine for the leakage audit, where we compare
        with-leak vs without-leak on the SAME labels, but NOT how you would build
        a training label in production (that comes in Milestone 3).
    """
    known = labels
    if as_of is not None:
        # Keep only chargebacks that had been reported on or before ``as_of``.
        known = labels[labels["reported_at"] <= as_of]

    fraud_codes = set(known["auth_code"].to_numpy())
    # ``.isin`` on a set is O(n): one hash lookup per transaction row.
    return transactions["auth_code"].isin(fraud_codes)
