"""Shared scoring/decision core -- the single source of truth for turning a
probability into a decision, used by BOTH the online service (M6) and the batch
scorer (M7).

Keeping this logic in one place is the whole point: if the online API and the
batch job derived the segment or the threshold even slightly differently, they
would hand back different decisions for the same transaction -- classic
train/serve (here, online/offline) skew. By routing both through these functions,
parity is guaranteed by construction, and ``test_online_batch_parity`` proves it.
"""

from __future__ import annotations

from collections.abc import Mapping

# entry-mode one-hots -> segment name, in the priority order used everywhere.
ENTRY_ONEHOTS = {
    "entry_CHIP": "CHIP",
    "entry_CONTACTLESS": "CONTACTLESS",
    "entry_SWIPE": "SWIPE",
    "entry_ECOM": "ECOM",
    "entry_MANUAL": "MANUAL",
}


def recover_segment(features: Mapping[str, float]) -> str | None:
    """Recover the entry-mode segment from the one-hot features (first set wins)."""
    for onehot, name in ENTRY_ONEHOTS.items():
        if float(features.get(onehot, 0.0)) >= 0.5:
            return name
    return None


def threshold_for(policy: dict | None, segment: str | None) -> float:
    """Per-segment threshold from the M5 policy (global fallback, then 0.5)."""
    if not policy:
        return 0.5
    seg_map = policy.get("segment_thresholds", {})
    if segment and segment in seg_map:
        return float(seg_map[segment])
    return float(policy.get("global_threshold", 0.5))


def decide(proba: float, threshold: float) -> str:
    """The single decision rule shared by every scoring path."""
    return "DECLINE" if proba >= threshold else "APPROVE"
