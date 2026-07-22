"""Merchant / MCC catalog and the chaotic-merchant-name generator.

Two jobs live here, both feeding the transaction generator:

1.  A small, realistic *merchant catalog*: each merchant has ONE clean
    ``merchant_id`` and ONE canonical name, plus its MCC (merchant category
    code) and a typical spend distribution. Clean ids are the "right answer" a
    downstream entity-resolution step is expected to recover.

2.  The *name corruptor*: real card-network descriptors are filthy. The same
    Walmart shows up as ``WAL-MART #1234``, ``WM SUPERCENTER DALLAS TX``,
    ``SQ *WALMART`` and so on. We pre-compute a pool of corrupted variants per
    merchant so the per-row assignment is a cheap array index even at 5M rows,
    while still guaranteeing "one merchant_id maps to many merchant_name
    strings" — the mess a naive ``GROUP BY merchant_name`` trips over.
"""

from __future__ import annotations

import numpy as np

# --- Honest, everyday merchants ------------------------------------------------
# (canonical_name, mcc, category, amount_mu, amount_sigma)  -- lognormal params.
_MERCHANTS: list[tuple[str, int, str, float, float]] = [
    ("WALMART", 5411, "grocery", 3.6, 0.8),
    ("KROGER", 5411, "grocery", 3.5, 0.7),
    ("H E B", 5411, "grocery", 3.5, 0.7),
    ("TOM THUMB", 5411, "grocery", 3.4, 0.7),
    ("TARGET", 5311, "department", 3.7, 0.8),
    ("COSTCO WHOLESALE", 5300, "warehouse", 4.6, 0.7),
    ("SAMS CLUB", 5300, "warehouse", 4.5, 0.7),
    ("WHATABURGER", 5814, "fast_food", 2.3, 0.5),
    ("CHICK FIL A", 5814, "fast_food", 2.4, 0.5),
    ("MCDONALDS", 5814, "fast_food", 2.1, 0.5),
    ("TORCHYS TACOS", 5812, "restaurant", 3.0, 0.5),
    ("PAPPADEAUX", 5812, "restaurant", 3.9, 0.6),
    ("STARBUCKS", 5814, "fast_food", 1.9, 0.5),
    ("SHELL OIL", 5541, "gas", 3.5, 0.6),
    ("EXXONMOBIL", 5541, "gas", 3.5, 0.6),
    ("QUIKTRIP", 5541, "gas", 3.3, 0.6),
    ("BEST BUY", 5732, "electronics", 4.6, 0.9),
    ("HOME DEPOT", 5200, "home_improve", 4.2, 0.9),
    ("LOWES", 5200, "home_improve", 4.2, 0.9),
    ("CVS PHARMACY", 5912, "pharmacy", 3.1, 0.7),
    ("WALGREENS", 5912, "pharmacy", 3.1, 0.7),
    ("AMAZON", 5942, "online_retail", 3.6, 0.9),
    ("NETFLIX", 5968, "subscription", 2.7, 0.3),
    ("SPOTIFY", 5968, "subscription", 2.3, 0.2),
    ("UBER", 4121, "rideshare", 2.9, 0.5),
    ("LYFT", 4121, "rideshare", 2.9, 0.5),
    ("AMERICAN AIRLINES", 4511, "airline", 5.6, 0.7),
    ("SOUTHWEST AIR", 4511, "airline", 5.4, 0.7),
    ("MARRIOTT", 3509, "lodging", 5.3, 0.6),
    ("ATT", 4814, "telecom", 4.0, 0.4),
    ("VERIZON", 4814, "telecom", 4.0, 0.4),
    ("ONCOR ELECTRIC", 4900, "utility", 4.2, 0.4),
    ("APPLE", 5732, "electronics", 4.3, 1.0),
    ("NIKE", 5651, "apparel", 4.0, 0.7),
    ("ROSS STORES", 5651, "apparel", 3.4, 0.6),
    ("PETSMART", 5995, "pet", 3.5, 0.7),
    ("REGAL CINEMAS", 7832, "entertainment", 3.2, 0.5),
    ("DICKS SPORTING", 5941, "sporting", 4.0, 0.7),
    ("CHEVRON", 5541, "gas", 3.5, 0.6),
    ("DOLLAR GENERAL", 5331, "variety", 2.7, 0.6),
]

# --- Ring merchants ------------------------------------------------------------
# The month-14 fraud ring funnels through a handful of high-risk digital-goods /
# quasi-cash merchants. These barely appear before onset, then spike -- a chunk
# of the concept drift the Milestone-8 monitor has to catch.
_RING_MERCHANTS: list[tuple[str, int, str, float, float]] = [
    ("QUICKCASH DIGITAL", 6051, "quasi_cash", 5.9, 0.4),
    ("GIFTCARD EXPRESS", 5816, "digital_goods", 5.8, 0.4),
    ("PRIME CRYPTO TOPUP", 6051, "quasi_cash", 6.0, 0.4),
    ("NOVA STREAMING STORE", 5816, "digital_goods", 5.7, 0.4),
]

# Store-number and city suffixes sprinkled into descriptors.
_CITIES: list[tuple[str, str]] = [
    ("DALLAS", "TX"), ("PLANO", "TX"), ("FRISCO", "TX"), ("AUSTIN", "TX"),
    ("HOUSTON", "TX"), ("IRVING", "TX"), ("ARLINGTON", "TX"), ("DENTON", "TX"),
]
# Payment-gateway / aggregator prefixes that mangle the real merchant name.
_GATEWAY_PREFIXES: list[str] = ["SQ *", "TST* ", "PP*", "SP ", "PAYPAL *", "TSYS*"]

_N_VARIANTS_PER_MERCHANT = 48  # size of each merchant's corrupted-name pool


def _strip_vowels(name: str) -> str:
    """Crude abbreviation: drop interior vowels the way tight descriptors do."""
    return name[0] + "".join(c for c in name[1:] if c.upper() not in "AEIOU ")


def _corrupt_once(rng: np.random.Generator, canonical: str) -> str:
    """Return ONE plausibly-mangled descriptor for ``canonical``.

    Applies a random subset of the transforms real POS/gateway systems inflict:
    casing, gateway prefixes, store numbers, city/state tails, vowel-stripping,
    truncation to the 25-char network limit, and stray whitespace.
    """
    s = canonical
    # 1) casing
    case_pick = rng.integers(0, 3)
    if case_pick == 0:
        s = s.upper()
    elif case_pick == 1:
        s = s.lower()
    else:
        s = s.title()

    # 2) optional vowel-stripped abbreviation
    if rng.random() < 0.20:
        s = _strip_vowels(s)

    # 3) optional gateway prefix
    if rng.random() < 0.30:
        s = _GATEWAY_PREFIXES[rng.integers(0, len(_GATEWAY_PREFIXES))] + s

    # 4) optional store number
    if rng.random() < 0.45:
        sep = rng.choice(["#", " #", " ", "-"])
        s = f"{s}{sep}{int(rng.integers(1, 9999)):04d}"

    # 5) optional city / state tail
    if rng.random() < 0.40:
        city, st = _CITIES[rng.integers(0, len(_CITIES))]
        s = f"{s} {city} {st}"

    # 6) network truncation to 25 chars
    s = s[:25]

    # 7) stray whitespace
    if rng.random() < 0.15:
        s = "  ".join(s.split(" ", 1)) if " " in s else s + " "

    return s


def build_catalog(rng: np.random.Generator) -> dict:
    """Build the full merchant catalog plus the pre-computed name-variant pool.

    Returns a dict with parallel numpy arrays indexed by ``merchant_idx``:
      - ``merchant_id``      : clean canonical id, e.g. ``M0007``
      - ``canonical``        : clean canonical name
      - ``mcc``              : merchant category code (int)
      - ``category``         : coarse category string
      - ``amt_mu``/``amt_sigma`` : lognormal spend params
      - ``is_ring``          : bool, True for the 4 ring merchants
      - ``name_variants``    : (n_merchants x _N_VARIANTS_PER_MERCHANT) object array
    """
    all_rows = _MERCHANTS + _RING_MERCHANTS
    n = len(all_rows)
    ring_start = len(_MERCHANTS)

    merchant_id = np.array([f"M{ i:04d}" for i in range(n)], dtype=object)
    canonical = np.array([r[0] for r in all_rows], dtype=object)
    mcc = np.array([r[1] for r in all_rows], dtype=np.int32)
    category = np.array([r[2] for r in all_rows], dtype=object)
    amt_mu = np.array([r[3] for r in all_rows], dtype=np.float64)
    amt_sigma = np.array([r[4] for r in all_rows], dtype=np.float64)
    is_ring = np.array([i >= ring_start for i in range(n)], dtype=bool)

    # Pre-compute the corrupted-name pool once per merchant. Per-transaction we
    # then just index into this, so 5M rows cost 5M array lookups, not 5M string
    # builds -- and every merchant still owns dozens of distinct descriptors.
    variants = np.empty((n, _N_VARIANTS_PER_MERCHANT), dtype=object)
    for m in range(n):
        for v in range(_N_VARIANTS_PER_MERCHANT):
            variants[m, v] = _corrupt_once(rng, str(canonical[m]))

    return {
        "merchant_id": merchant_id,
        "canonical": canonical,
        "mcc": mcc,
        "category": category,
        "amt_mu": amt_mu,
        "amt_sigma": amt_sigma,
        "is_ring": is_ring,
        "name_variants": variants,
        "n_variants": _N_VARIANTS_PER_MERCHANT,
        "ring_start": ring_start,
    }
