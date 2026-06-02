"""BM25 retrieval helpers."""

from __future__ import annotations


def bm25_pipeline(index_ref: str):
    """Return a simple PyTerrier BM25 retriever for an index reference."""
    import pyterrier as pt

    return pt.BatchRetrieve(index_ref, wmodel="BM25")

