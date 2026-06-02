"""Cluster-Hypothesis reranking transformer compatible with PyTerrier pipelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ClusterInterceptor:
    """Diversity-aware reranker that rewards unseen pseudo-clusters per query."""

    penalty: float = 0.05
    n_clusters: int = 16

    def _cluster_id(self, doc_identifier: str) -> int:
        digest = hashlib.sha1(doc_identifier.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.n_clusters

    def _rerank(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        ranked = frame.sort_values(["qid", "score"], ascending=[True, False]).copy()
        ranked["cluster_id"] = ranked["docno"].astype(str).map(self._cluster_id)

        adjusted_scores: list[float] = []
        for _, query_slice in ranked.groupby("qid", sort=False):
            seen_clusters: dict[int, int] = {}
            for row in query_slice.itertuples(index=False):
                cluster = int(row.cluster_id)
                previous_hits = seen_clusters.get(cluster, 0)
                adjusted_scores.append(float(row.score) - self.penalty * previous_hits)
                seen_clusters[cluster] = previous_hits + 1

        ranked["score"] = adjusted_scores
        ranked = ranked.sort_values(["qid", "score"], ascending=[True, False])
        ranked["rank"] = ranked.groupby("qid").cumcount() + 1
        return ranked.drop(columns=["cluster_id"])

    def as_transformer(self) -> Any:
        """Return a `pt.apply.generic` transformer suitable for `>>` pipelines."""
        try:
            import pyterrier as pt
        except ImportError as error:
            raise ImportError(
                "PyTerrier is required to build a pipeline transformer."
            ) from error

        return pt.apply.generic(self._rerank)


def build_cluster_transformer(penalty: float = 0.05, n_clusters: int = 16) -> Any:
    """Factory for PyTerrier pipeline usage: `retriever >> build_cluster_transformer()`."""
    return ClusterInterceptor(penalty=penalty, n_clusters=n_clusters).as_transformer()

