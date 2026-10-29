"""Deterministic seed derivation: every draw comes from (run seed, episode id, ...)."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(*keys: object) -> int:
    digest = hashlib.sha256("|".join(repr(k) for k in keys).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def derive_rng(*keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*keys))
