"""The three PLANTED leakage traps.

Data leakage = a feature that secretly carries information no honest model could
have had at scoring time. We plant three, each a *different archetype*, so the
Milestone-2 leakage audit has to catch three different failure modes rather than
one. They are deliberately tempting: each looks like a normal operational column
a hurried modeller would happily ``SELECT``.

    LEAK A -- ``dispute_reason_code``       : a post-outcome column. Only exists
              because the chargeback already happened. Perfectly separates
              classes. Archetype: *label smuggled in as a raw field.*

    LEAK B -- ``merchant_fraud_rate_lifetime`` : a target aggregate computed over
              the merchant's WHOLE history (past AND future), so it embeds the
              very labels being predicted -- including for transactions that
              happened before the fraud did. Archetype: *look-ahead target
              encoding.*

    LEAK C -- ``card_txn_count_next_24h``  : counts the card's transactions in the
              24h AFTER this one. A future window. Archetype: *temporal
              look-ahead feature.*

None of these should survive the audit. The registry below is what the audit
(and the tests) key off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Registry the audit + tests import. Keep in sync with the functions below.
LEAK_COLUMNS: dict[str, str] = {
    "dispute_reason_code": "post_outcome_field",
    "merchant_fraud_rate_lifetime": "lookahead_target_encoding",
    "card_txn_count_next_24h": "temporal_lookahead_window",
}

# Visa/Mastercard-style fraud dispute reason codes. In the real world these are
# only assigned once a chargeback is filed -- i.e. strictly after the label.
_FRAUD_REASON_CODES = np.array(["10.4", "10.1", "10.5", "13.1", "4837"], dtype=object)
_NO_DISPUTE = "NONE"


def add_leak_a_dispute_code(
    rng: np.random.Generator, is_fraud_row: np.ndarray
) -> np.ndarray:
    """LEAK A -- a dispute reason code present iff the transaction is fraud.

    ``is_fraud_row`` is a boolean array aligned to the transactions table. Fraud
    rows get a real-looking reason code; everything else gets ``"NONE"``. Because
    presence is a perfect function of the label, any model that reads this column
    is reading the answer key.
    """
    codes = np.full(is_fraud_row.shape[0], _NO_DISPUTE, dtype=object)
    fraud_idx = np.flatnonzero(is_fraud_row)
    codes[fraud_idx] = _FRAUD_REASON_CODES[
        rng.integers(0, len(_FRAUD_REASON_CODES), size=fraud_idx.shape[0])
    ]
    return codes


def add_leak_b_merchant_lifetime_rate(
    merchant_id: np.ndarray, is_fraud_row: np.ndarray
) -> np.ndarray:
    """LEAK B -- each merchant's fraud rate over ALL time, stamped on every row.

    Computed with a global ``groupby`` -- no ``as_of`` cut -- so the value on a
    January transaction already reflects fraud that only happens in December.
    That look-ahead is the leak: honest feature code would compute this rate only
    from data available *before* each transaction.
    """
    frame = pd.DataFrame({"merchant_id": merchant_id, "is_fraud": is_fraud_row.astype(float)})
    lifetime = frame.groupby("merchant_id")["is_fraud"].transform("mean")
    return lifetime.to_numpy()


def add_leak_c_next_24h_count(
    card_id: np.ndarray, event_ts: np.ndarray
) -> np.ndarray:
    """LEAK C -- count of the same card's transactions in the next 24 hours.

    A strictly forward-looking window. For each row at time ``t`` we count that
    card's transactions in ``(t, t + 24h]``. An honest velocity feature would use
    the *past* 24h; using the future is the plant.

    Implemented by sorting once by (card, time) and walking contiguous same-card
    blocks with ``searchsorted`` -- O(n log n), fine at 5M rows.
    """
    n = card_id.shape[0]
    out = np.zeros(n, dtype=np.int32)

    ts_ns = event_ts.astype("datetime64[ns]").astype(np.int64)
    horizon = np.int64(24 * 3600) * np.int64(1_000_000_000)  # 24h in nanoseconds

    # Stable sort by card, then time. lexsort keys are least-significant-first.
    order = np.lexsort((ts_ns, card_id))
    card_sorted = card_id[order]
    ts_sorted = ts_ns[order]

    # Block boundaries = where the card id changes in the sorted view.
    change = np.flatnonzero(card_sorted[1:] != card_sorted[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))

    for s, e in zip(starts, ends, strict=True):
        block = ts_sorted[s:e]  # already time-sorted within the card block
        # first index with time strictly greater than t  -> excludes t itself
        lower = np.searchsorted(block, block, side="right")
        # first index with time strictly greater than t + 24h
        upper = np.searchsorted(block, block + horizon, side="right")
        out[order[s:e]] = (upper - lower).astype(np.int32)

    return out
