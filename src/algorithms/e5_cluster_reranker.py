"""
E5ClusterReranker: cluster-based re-ranking using dense bi-encoder embeddings.

Two clustering modes:
  mode='knn'    — each passage is the centre of its k nearest neighbours (overlapping)
  mode='kmeans' — K-means partitions into k disjoint clusters

Three centroid types for cluster scoring (centroid · query similarity):
  centroid_type='mean'          — arithmetic mean of member embeddings, L2-normalised
  centroid_type='medoid'        — member embedding closest to the arithmetic mean
  centroid_type='query_weighted'— mean weighted by softmax(passage–query cosine sims)

Clusters are ranked by centroid–query cosine similarity (highest first).
Within each cluster, passages are ranked by individual passage–query cosine similarity.
Scores are assigned as descending integers to preserve ordering for interpolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

_VALID_MODES    = ('knn', 'kmeans')
_VALID_CENTROIDS = ('mean', 'medoid', 'query_weighted')


class E5ClusterReranker:
    """
    Re-ranks passages by clustering their dense bi-encoder embeddings.

    Parameters
    ----------
    model         : HuggingFace AutoModel.
    tokenizer     : Matching AutoTokenizer.
    k             : Neighbourhood size (knn) or number of clusters (kmeans).
    mode          : 'knn' (default) or 'kmeans'.
    centroid_type : How to compute the cluster representative used for scoring.
                    'mean' (default) — arithmetic mean, L2-normalised.
                    'medoid'         — member whose embedding is closest to the mean.
                    'query_weighted' — mean weighted by softmax(passage-query sims).
    query_prefix  : Prefix prepended to the query string before encoding.
    doc_prefix    : Prefix prepended to each passage string before encoding.
    encode_batch  : Tokenisation batch size.
    device        : 'cuda' | 'cpu' | None (auto-detect).
    """

    def __init__(
        self,
        model,
        tokenizer,
        k: int = 5,
        mode: str = 'knn',
        centroid_type: str = 'mean',
        query_prefix: str = 'query: ',
        doc_prefix: str = 'passage: ',
        encode_batch: int = 64,
        device: str | None = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}")
        if centroid_type not in _VALID_CENTROIDS:
            raise ValueError(f"centroid_type must be one of {_VALID_CENTROIDS}")
        self.tokenizer     = tokenizer
        self.k             = k
        self.mode          = mode
        self.centroid_type = centroid_type
        self.query_prefix  = query_prefix
        self.doc_prefix    = doc_prefix
        self.encode_batch  = encode_batch
        self.device        = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model         = model.to(self.device).eval()

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
        sim = d_vecs @ d_vecs.T
        k   = min(self.k, len(d_vecs))
        return [np.argsort(sim[i])[::-1][:k] for i in range(len(d_vecs))]

    def _build_kmeans_clusters(self, d_vecs: np.ndarray) -> list[np.ndarray]:
        """Partition into k disjoint K-means clusters."""
        from sklearn.cluster import KMeans
        k_actual = min(self.k, len(d_vecs))
        labels   = KMeans(n_clusters=k_actual, random_state=42, n_init=10).fit_predict(d_vecs)
        return [np.where(labels == cid)[0] for cid in range(k_actual)]

    def _centroid(
        self,
        members: np.ndarray,
        d_vecs: np.ndarray,
        passage_sims: np.ndarray,
    ) -> np.ndarray:
        """Return a unit-vector representative for the cluster."""
        if self.centroid_type == 'mean':
            c    = d_vecs[members].mean(axis=0)
            norm = np.linalg.norm(c)
            return c / norm if norm > 1e-9 else c

        if self.centroid_type == 'medoid':
            mean_c = d_vecs[members].mean(axis=0)
            dists  = np.linalg.norm(d_vecs[members] - mean_c, axis=1)
            return d_vecs[members[np.argmin(dists)]]   # already unit-norm

        # query_weighted: softmax of passage–query cosines as mixing weights
        sims = passage_sims[members].astype(float)
        w    = np.exp(sims - sims.max())
        w   /= w.sum()
        c    = (d_vecs[members] * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(c)
        return c / norm if norm > 1e-9 else c

    def _rerank_query(self, query_text: str, group: pd.DataFrame) -> pd.DataFrame:
        group = group.reset_index(drop=True)
        docs  = group['text'].fillna('').tolist()

        q_vec        = self._encode([self.query_prefix + query_text])
        d_vecs       = self._encode_all([self.doc_prefix + t for t in docs])
        passage_sims = (d_vecs @ q_vec.T).squeeze(-1)

        clusters = (self._build_knn_clusters(d_vecs)
                    if self.mode == 'knn'
                    else self._build_kmeans_clusters(d_vecs))

        q = q_vec[0]
        centroid_sims = [
            float(self._centroid(members, d_vecs, passage_sims) @ q)
            for members in clusters
        ]
        cluster_order = np.argsort(centroid_sims)[::-1]

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
        for qid, group in tqdm(
            run_df.groupby('qid'),
            desc=f'ClusterRerank ({self.mode}, {self.centroid_type})',
            leave=True,
        ):
            results.append(self._rerank_query(str(group[query_col].iloc[0]), group))
        return pd.concat(results, ignore_index=True)
