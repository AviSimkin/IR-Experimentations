"""
ClustMRFSVMRank: learn ClustMRF feature weights via SVMrank with k-fold CV.

Training procedure (Raiber & Kurland §3.1):
  For each query q in the training set:
    For each pair (di, dj) from the top-n_docs documents where rel(di) > rel(dj):
      Add feature-difference vector  fi − fj  with label +1
      Add feature-difference vector  fj − fi  with label −1
  Train sklearn LinearSVC (= SVMrank with L2 regularisation, C=self.C).

Cross-validation:
  Queries are split into n_folds groups (KFold, no shuffle, deterministic).
  Each fold is used once as the test set; the model is trained on the remaining
  folds.  The final run DataFrame covers every query exactly once, so evaluation
  against the full qrel set is unbiased and directly comparable to heuristic-
  weight ClustMRF evaluated on the same query set.

  Nested CV (for tuning C) is NOT implemented: fixing C=1.0 matches common IR
  practice and avoids the cost of an inner loop.  If you want to tune C, wrap
  fit_transform_cv in an outer LOO loop and call sklearn.model_selection.GridSearchCV
  inside each iteration on the training fold only.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

from src.algorithms.clust_mrf import ClustMRF, FEATURE_NAMES


class ClustMRFSVMRank:
    """ClustMRF with SVMrank-trained weights, evaluated via k-fold CV.

    Parameters
    ----------
    index : Terrier index reference
        Passed through to ClustMRF (kept for API consistency).
    k : int
        Cluster size (number of nearest neighbours + the centre document).
    n_docs : int
        Size of the initial re-ranking pool (paper's D_init).
    C : float
        SVM regularisation parameter (sklearn LinearSVC C).  Higher = less
        regularisation / harder margin.  Default 1.0 matches SVMrank default.
    n_folds : int
        Number of cross-validation folds.  5-fold is standard in TREC IR papers.
    n_jobs : int
        Parallelism passed to ClustMRF.transform_features() for feature extraction.
    """

    def __init__(
        self,
        index,
        k: int = 5,
        n_docs: int = 50,
        C: float = 1.0,
        n_folds: int = 5,
        n_jobs: int = -1,
    ):
        self.index   = index
        self.k       = k
        self.n_docs  = n_docs
        self.C       = C
        self.n_folds = n_folds
        self.n_jobs  = n_jobs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform_cv(
        self,
        init_run: pd.DataFrame,
        qrels_df: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run n-fold CV: train SVMrank on held-out training queries, predict on
        test queries.  Returns a full run DataFrame covering all queries.

        Parameters
        ----------
        init_run : DataFrame
            Initial retrieval run with columns qid, query, docno, rank, score, text.
        qrels_df : pd.DataFrame
            Relevance judgments with columns query_id, doc_id, relevance.
        verbose : bool
            Print per-fold timing.

        Returns
        -------
        pd.DataFrame with columns qid, docno, rank, score (ClustMRF score).
        """
        # ── 1. Extract features for every query (once, outside the fold loop) ──
        cmrf = ClustMRF(index=self.index, k=self.k, n_docs=self.n_docs,
                        n_jobs=self.n_jobs)
        if verbose:
            print(f'Extracting ClustMRF features (k={self.k}, n_docs={self.n_docs})...')
        t0 = time.time()

        # Store full extracted dicts (features + clustering structure) per query.
        # We need cluster_nn and sim_qd to do the cluster-unrolling step after
        # applying the SVM weight vector at test time.
        raw_groups = {
            qid: grp.reset_index(drop=True)
            for qid, grp in init_run.groupby('qid', sort=False)
        }
        all_extracted: dict[str, dict] = {}
        for qid, grp in raw_groups.items():
            ext = cmrf._extract_features_for_query(grp)
            if ext is not None:
                ext['query'] = str(grp['query'].iloc[0]) if 'query' in grp.columns else ''
                all_extracted[qid] = ext

        if verbose:
            print(f'  {len(all_extracted)} queries extracted in {time.time()-t0:.1f}s')

        # ── 2. Build relevance lookup ─────────────────────────────────────────
        qrel_lookup: dict[str, dict[str, int]] = defaultdict(dict)
        for _, row in qrels_df.iterrows():
            qrel_lookup[str(row['query_id'])][str(row['doc_id'])] = int(row['relevance'])

        # ── 3. k-fold CV ──────────────────────────────────────────────────────
        qids = sorted(all_extracted.keys())
        kf   = KFold(n_splits=self.n_folds, shuffle=False)
        all_results: list[pd.DataFrame] = []
        learned_weights: list[np.ndarray] = []

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(qids)):
            train_qids = [qids[i] for i in train_idx]
            test_qids  = [qids[i] for i in test_idx]

            # ── 3a. Build pairwise training data ──────────────────────────────
            X_pairs: list[np.ndarray] = []
            y_pairs: list[int]        = []

            for qid in train_qids:
                ext    = all_extracted[qid]
                feats  = ext['features']                   # (n, 19)
                docnos = ext['top']['docno'].tolist()
                rels   = np.array(
                    [qrel_lookup[qid].get(d, 0) for d in docnos], dtype=int
                )

                # All ordered pairs (i, j) where rel_i > rel_j within top-n
                for i in range(len(docnos)):
                    for j in range(i + 1, len(docnos)):
                        if rels[i] > rels[j]:
                            diff = feats[i] - feats[j]
                        elif rels[j] > rels[i]:
                            diff = feats[j] - feats[i]
                        else:
                            continue
                        # Add both directions for balanced classes
                        X_pairs.append(diff);  y_pairs.append(+1)
                        X_pairs.append(-diff); y_pairs.append(-1)

            if not X_pairs:
                if verbose:
                    print(f'  Fold {fold_i+1}/{self.n_folds}: no training pairs — skipping')
                continue

            X_tr = np.vstack(X_pairs)
            y_tr = np.array(y_pairs, dtype=int)

            # ── 3b. Train LinearSVC (= SVMrank) ───────────────────────────────
            t1 = time.time()
            svm = LinearSVC(C=self.C, max_iter=10_000, dual='auto')
            svm.fit(X_tr, y_tr)
            weights = svm.coef_[0]   # (19,)
            learned_weights.append(weights)

            if verbose:
                n_pos = int((y_tr == 1).sum())
                print(f'  Fold {fold_i+1}/{self.n_folds}: '
                      f'{len(train_qids)} train / {len(test_qids)} test queries, '
                      f'{n_pos:,} pairs, '
                      f'SVM fit {time.time()-t1:.1f}s')

            # ── 3c. Score + unroll test queries with learned weights ───────────
            for qid in test_qids:
                ext = all_extracted[qid]
                ranked = cmrf._apply_weights_and_unroll(ext, weights)
                ranked = ranked[['docno', 'score', 'rank']].copy()
                ranked['qid'] = qid
                all_results.append(ranked)

        if not all_results:
            return pd.DataFrame(columns=['qid', 'docno', 'rank', 'score'])

        result_df = pd.concat(all_results, ignore_index=True)
        # Re-compute ranks within each query from the continuous cluster scores
        result_df['rank'] = (
            result_df.groupby('qid')['score']
            .rank(ascending=False, method='first')
            .astype(int)
        )

        if verbose and learned_weights:
            mean_w = np.mean(learned_weights, axis=0)
            top3   = np.argsort(np.abs(mean_w))[::-1][:3]
            top3_str = ', '.join(f'{FEATURE_NAMES[i]}={mean_w[i]:+.3f}' for i in top3)
            print(f'  Mean weights (top-3 by |w|): {top3_str}')

        return result_df

    # ------------------------------------------------------------------
    # Convenience: return the per-fold weight vectors for analysis
    # ------------------------------------------------------------------

    def fit_cv_weights(
        self,
        init_run: pd.DataFrame,
        qrels_df: pd.DataFrame,
        verbose: bool = True,
    ) -> np.ndarray:
        """Like fit_transform_cv but returns the (n_folds, 19) weight matrix
        instead of the re-ranked run.  Useful for inspecting what SVMrank learns."""
        cmrf = ClustMRF(index=self.index, k=self.k, n_docs=self.n_docs,
                        n_jobs=self.n_jobs)
        raw_groups = {
            qid: grp.reset_index(drop=True)
            for qid, grp in init_run.groupby('qid', sort=False)
        }
        all_extracted = {}
        for qid, grp in raw_groups.items():
            ext = cmrf._extract_features_for_query(grp)
            if ext is not None:
                all_extracted[qid] = ext

        qrel_lookup: dict[str, dict[str, int]] = defaultdict(dict)
        for _, row in qrels_df.iterrows():
            qrel_lookup[str(row['query_id'])][str(row['doc_id'])] = int(row['relevance'])

        qids = sorted(all_extracted.keys())
        kf   = KFold(n_splits=self.n_folds, shuffle=False)
        weight_matrix: list[np.ndarray] = []

        for fold_i, (train_idx, _) in enumerate(kf.split(qids)):
            train_qids = [qids[i] for i in train_idx]
            X_pairs, y_pairs = [], []
            for qid in train_qids:
                ext   = all_extracted[qid]
                feats = ext['features']
                docnos = ext['top']['docno'].tolist()
                rels   = np.array([qrel_lookup[qid].get(d, 0) for d in docnos], dtype=int)
                for i in range(len(docnos)):
                    for j in range(i + 1, len(docnos)):
                        if rels[i] > rels[j]:
                            diff = feats[i] - feats[j]
                        elif rels[j] > rels[i]:
                            diff = feats[j] - feats[i]
                        else:
                            continue
                        X_pairs.append(diff);  y_pairs.append(+1)
                        X_pairs.append(-diff); y_pairs.append(-1)
            if not X_pairs:
                continue
            svm = LinearSVC(C=self.C, max_iter=10_000, dual='auto')
            svm.fit(np.vstack(X_pairs), np.array(y_pairs, dtype=int))
            weight_matrix.append(svm.coef_[0])
            if verbose:
                print(f'  Fold {fold_i+1}: trained')

        return np.array(weight_matrix)
