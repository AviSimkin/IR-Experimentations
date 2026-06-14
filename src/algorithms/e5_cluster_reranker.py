"""
E5ClusterReranker: cluster-based re-ranking using E5 bi-encoder embeddings.

Two clustering modes, both sharing the same ranking rule:
  rank clusters by centroid–query cosine similarity (closest first);
  within each cluster rank passages by individual passage–query cosine similarity;
  flatten and skip duplicate docnos.

mode='knn'  (default)
  Each passage is the centre of a cluster containing its k-1 nearest neighbours
  (by E5 cosine similarity) — exactly like ClustMRF but with dense vectors instead
  of TF-IDF.  Produces n overlapping clusters of size k; duplicates resolved by
  first-seen rule during unrolling.

mode='kmeans'
  K-means partitions passages into k disjoint clusters.  No duplicate docnos.

Scores are assigned as descending integers (n, n-1, …, 1) to preserve cluster-
priority ordering for downstream interpolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


class E5ClusterReranker:
    """
    Re-ranks passages by clustering their E5 embeddings.

    Parameters
    ----------
    model        : HuggingFace AutoModel (e5-base-v2 or compatible).
    tokenizer    : Matching AutoTokenizer.
    k            : Neighbourhood size (knn) or number of clusters (kmeans).
    mode         : 'knn' (default) or 'kmeans'.
    encode_batch : Tokenisation batch size.
    device       : 'cuda' | 'cpu' | None (auto-detect).
    """

    def __init__(self, model, tokenizer, k: int = 5, mode: str = 'knn',
                 encode_batch: int = 64, device: str | None = None):
        if mode not in ('knn', 'kmeans'):
            raise ValueError("mode must be 'knn' or 'kmeans'")
        self.tokenizer    = tokenizer
        self.k            = k
        self.mode         = mode
        self.encode_batch = encode_batch
        self.device       = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = model.to(self.device).eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors='pt',
        ).to(self.device)
        with torch.inference_mode():
            out = self.model(**enc)
        mask   = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1).cpu().float().numpy()

    def _encode_all(self, texts: list[str]) -> np.ndarray:
        return np.vstack([
            self._encode(texts[i:i + self.encode_batch])
            for i in range(0, len(texts), self.encode_batch)
        ])

    def _build_knn_clusters(self, d_vecs: np.ndarray) -> list[np.ndarray]:
        """Each doc → cluster of its k nearest neighbours (including itself)."""
        sim = d_vecs @ d_vecs.T   # (n, n) — already L2-normalised
        k   = min(self.k, len(d_vecs))
        return [np.argsort(sim[i])[::-1][:k] for i in range(len(d_vecs))]

    def _build_kmeans_clusters(self, d_vecs: np.ndarray) -> list[np.ndarray]:
        """Partition into k disjoint K-means clusters."""
        from sklearn.cluster import KMeans
        k_actual = min(self.k, len(d_vecs))
        labels   = KMeans(n_clusters=k_actual, random_state=42, n_init=10).fit_predict(d_vecs)
        return [np.where(labels == cid)[0] for cid in range(k_actual)]

    def _rerank_query(self, query_text: str, group: pd.DataFrame) -> pd.DataFrame:
        group = group.reset_index(drop=True)
        docs  = group['text'].fillna('').tolist()

        q_vec  = self._encode(['query: ' + query_text])              # (1, d)
        d_vecs = self._encode_all(['passage: ' + t for t in docs])   # (n, d)

        passage_sims = (d_vecs @ q_vec.T).squeeze(-1)   # (n,)

        clusters = (self._build_knn_clusters(d_vecs)
                    if self.mode == 'knn'
                    else self._build_kmeans_clusters(d_vecs))

        # Score each cluster by normalised-centroid · query
        q = q_vec[0]                         # (d,) — avoid shape issues with q_vec.T
        centroid_sims = []
        for members in clusters:
            c     = d_vecs[members].mean(axis=0)
            norm  = np.linalg.norm(c)
            c_hat = c / norm if norm > 1e-9 else c
            centroid_sims.append(float(c_hat @ q))

        cluster_order = np.argsort(centroid_sims)[::-1]

        # Unroll: best cluster first; within cluster best passage-query sim first;
        # skip already-placed docnos (critical for knn where clusters overlap).
        ranked_indices: list[int] = []
        seen_docnos: set[str]     = set()
        for ci in cluster_order:
            members = clusters[ci]
            for idx in members[np.argsort(passage_sims[members])[::-1]]:
                docno = group.at[int(idx), 'docno']
                if docno not in seen_docnos:
                    seen_docnos.add(docno)
                    ranked_indices.append(int(idx))

        out          = group.iloc[ranked_indices].copy()
        out['rank']  = range(1, len(out) + 1)
        out['score'] = range(len(out), 0, -1)
        return out

    def transform(self, run_df: pd.DataFrame) -> pd.DataFrame:
        query_col = 'query_0' if 'query_0' in run_df.columns else 'query'
        results   = []
        for qid, group in tqdm(run_df.groupby('qid'),
                               desc=f'E5-cluster rerank ({self.mode})', leave=True):
            results.append(self._rerank_query(str(group[query_col].iloc[0]), group))
        return pd.concat(results, ignore_index=True)
