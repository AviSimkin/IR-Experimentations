"""Dataset streaming helpers for TREC/MS MARCO collections."""

from __future__ import annotations

from typing import Any

import ir_datasets


def load_dataset(dataset_id: str) -> Any:
    """Load an ir_datasets dataset handle."""
    return ir_datasets.load(dataset_id)

