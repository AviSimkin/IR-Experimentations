"""Metric definitions and evaluation helpers."""

from __future__ import annotations

from typing import Iterable

import ir_measures
from ir_measures import MAP, MRR, nDCG


def default_measures(k: int = 10) -> Iterable[ir_measures.Measure]:
    """Return baseline ranking measures used by this project."""
    return (MAP, nDCG@k, MRR@k)

