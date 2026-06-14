"""
ClustMRF: Ranking document clusters using Markov Random Fields.
(Raiber & Kurland, SIGIR 2013)

score(C, Q) = Σ_{l∈L(G)} λ_l · f_l(l)                             [Eq. 3]

Three clique types (Fig. 1):

  lQD  (query + one document):
       f_geo_qsim = (1/|C|) log sim(Q, d)  — summing over all lQD cliques
                                              yields the geometric mean of
                                              query-similarity values.

  lQC  (query + all cluster documents):
       f_min_qsim  = log  min_{d∈C} sim(Q, d)
       f_max_qsim  = log  max_{d∈C} sim(Q, d)
       f_stdv_qsim = log stddev({sim(Q,d)}_{d∈C})

  lC   (cluster documents only — query-independent):
       For each document measure P, three aggregators A ∈ {geo, min, max}:
         f_{A-P}(C) = log A({P(d)}_{d∈C})

       Non-web document measures:
         P_dsim(d)      = (1/|C|) Σ_{di∈C} cos_sim(d, di)  [cluster coherence]
         P_entropy(d)   = −Σ_w p(w|d) log p(w|d)            [content breadth]
         P_icompress(d) = |d|_bytes / |gzip(d)|_bytes        [content breadth]

       Web-specific document measures (ClueWeb09 collections):
         P_spam(d)    = Waterloo spam score ∈ [0,1] (1 = definitely not spam)
         P_pr(d)      = PageRank score (raw; log-transformed internally)
         P_urlslash(d)= number of slashes in the URL path (URL depth)
         P_urllen(d)  = URL character length

sim(Q, d)  = initial retrieval score, softmax-normalised to (0, 1].
cos_sim    = cosine similarity from stemmed TF-IDF vectors.
log uses add-ε (= 1e-10) smoothing before application (paper footnote 1).

Non-web default weights are proportional to the feature-importance order
reported in Table 3 (non-web setting) of Raiber & Kurland 2013.
Web feature weights default to 0.0 (inactive); set them in the constructor
when using ClueWeb09 collections.  All active weights should sum to 1.0.

Cluster → document ranking (paper §4.1):
  Clusters are sorted best→worst.  For each cluster, constituent documents
  are appended to the final list in descending sim(Q,d) order, skipping
  documents already placed by a higher-scoring cluster.
"""

from __future__ import annotations

import math
import os
import re
import gzip as _gzip
import collections
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

EPS = 1e-10   # add-ε smoothing before log (paper footnote 1)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Porter Stemmer (M.F. Porter, 1980) — aligns with Terrier's PorterStemmer
# ---------------------------------------------------------------------------

_VOWELS = frozenset("aeiou")

def _has_vowel(s): return any(c in _VOWELS for c in s)
def _ends_double_consonant(w): return len(w) >= 2 and w[-1] == w[-2] and w[-1] not in _VOWELS
def _ends_cvc(w):
    return (len(w) >= 3 and w[-1] not in _VOWELS and w[-2] in _VOWELS
            and w[-3] not in _VOWELS and w[-1] not in "wxy")

def _measure(s):
    n, prev_v = 0, False
    for c in s:
        if c in _VOWELS:
            prev_v = True
        elif prev_v:
            n += 1; prev_v = False
    return n

def porter_stem(word: str) -> str:
    if len(word) <= 2:
        return word
    w = word.lower()
    if w.endswith("sses"):   w = w[:-2]
    elif w.endswith("ies"):  w = w[:-2]
    elif w.endswith("ss"):   pass
    elif w.endswith("s"):    w = w[:-1]

    if w.endswith("eed"):
        if _measure(w[:-3]) > 0: w = w[:-1]
    elif w.endswith("ed"):
        s = w[:-2]
        if _has_vowel(s):
            w = s
            if w.endswith(("at","bl","iz")):            w += "e"
            elif _ends_double_consonant(w) and w[-1] not in "lsz": w = w[:-1]
            elif _measure(w) == 1 and _ends_cvc(w):    w += "e"
    elif w.endswith("ing"):
        s = w[:-3]
        if _has_vowel(s):
            w = s
            if w.endswith(("at","bl","iz")):            w += "e"
            elif _ends_double_consonant(w) and w[-1] not in "lsz": w = w[:-1]
            elif _measure(w) == 1 and _ends_cvc(w):    w += "e"

    if w.endswith("y") and _has_vowel(w[:-1]): w = w[:-1] + "i"

    for sfx, rep in [("ational","ate"),("tional","tion"),("enci","ence"),
                     ("anci","ance"),("izer","ize"),("abli","able"),("alli","al"),
                     ("entli","ent"),("eli","e"),("ousli","ous"),("ization","ize"),
                     ("ation","ate"),("ator","ate"),("alism","al"),("iveness","ive"),
                     ("fulness","ful"),("ousness","ous"),("aliti","al"),("iviti","ive"),
                     ("biliti","ble")]:
        if w.endswith(sfx) and _measure(w[:-len(sfx)]) > 0:
            w = w[:-len(sfx)] + rep; break

    for sfx, rep in [("icate","ic"),("ative",""),("alize","al"),("iciti","ic"),
                     ("ical","ic"),("ful",""),("ness","")]:
        if w.endswith(sfx) and _measure(w[:-len(sfx)]) > 0:
            w = w[:-len(sfx)] + rep; break

    for sfx in ["al","ance","ence","er","ic","able","ible","ant","ement",
                "ment","ent","ion","ou","ism","ate","iti","ous","ive","ize"]:
        if w.endswith(sfx):
            stem = w[:-len(sfx)]
            if _measure(stem) > 1:
                if sfx == "ion" and stem and stem[-1] in "st": w = stem
                elif sfx != "ion": w = stem
            break

    if w.endswith("e"):
        s = w[:-1]
        if _measure(s) > 1: w = s
        elif _measure(s) == 1 and not _ends_cvc(s): w = s
    if _ends_double_consonant(w) and w.endswith("l") and _measure(w[:-1]) > 1:
        w = w[:-1]
    return w


def tokenize_and_stem(text: str) -> list[str]:
    return [porter_stem(t) for t in tokenize(text)]


# ---------------------------------------------------------------------------
# Per-document measures (lC clique, query-independent)
# ---------------------------------------------------------------------------

def _entropy(tokens: list[str]) -> float:
    """P_entropy: unsmoothed unigram entropy = −Σ p(w|d) log p(w|d)."""
    if not tokens:
        return 0.0
    n = len(tokens)
    tf = collections.Counter(tokens)
    return -sum((c / n) * math.log(c / n) for c in tf.values())


def _icompress(text: str) -> float:
    """P_icompress: inverse gzip compression ratio = len(raw) / len(gzip(raw)).
    Higher value = harder to compress = richer content."""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(raw) / max(len(_gzip.compress(raw, compresslevel=6)), 1)


def url_features(url: str) -> dict[str, float]:
    """
    Compute P_urlslash and P_urllen from a document URL.
    P_urlslash: number of slashes in the URL path (URL depth proxy).
    P_urllen:   total URL character length.
    Both are raw counts; ClustMRF applies log(x + ε) like all lC features.
    """
    if not url:
        return {"urlslash": 0.0, "urllen": 0.0}
    path = url.split("?")[0].split("#")[0]   # strip query/fragment
    n_slash = max(0, path.count("/") - 2)    # subtract the two in http://
    return {"urlslash": float(n_slash), "urllen": float(len(url))}


# ---------------------------------------------------------------------------
# ClustMRF PyTerrier Transformer
# ---------------------------------------------------------------------------

try:
    import pyterrier as pt
    _HAS_PT = True
except ImportError:
    _HAS_PT = False


class ClustMRF:
    """
    ClustMRF re-ranker (Raiber & Kurland, SIGIR 2013).

    Input DataFrame must contain: qid, query, docno, rank, score, text.
    The 'text' column must be populated (use pt.text.get_text() upstream).

    Parameters
    ----------
    index : Terrier index reference (kept for API compatibility; not used for
            scoring — all features derive from the input text and scores).
    k : int
        Cluster size. Each document d is the center of C(d) = {d} ∪ {k-1
        nearest neighbours by TF-IDF cosine similarity within D_init}.
    n_docs : int
        |D_init|: how many top-ranked docs to re-rank (rest appended after).
    doc_features : dict[str, dict] | None
        Optional per-document features for web collections:
          {docno: {'spam': float,     # Waterloo spam score ∈ [0,1]
                   'pr':   float,     # raw PageRank (log-transformed internally)
                   'urlslash': float, # URL path slash count
                   'urllen':  float}} # URL character length
        Missing docnos receive neutral defaults (spam=0.5, pr=0, url*=0).
        Set the corresponding weight params (w_geo_spam etc.) to non-zero
        values to activate web features.
    w_geo_qsim … w_max_entropy : float
        Feature weights for the 13 non-web features.
        Default values proportional to Table 3 (non-web setting, SIGIR 2013).
    w_geo_spam … w_max_spam : float
        lC weights for P_spam (Waterloo spam score).  Default 0.0 (inactive).
    w_geo_pr … w_max_pr : float
        lC weights for P_pr (PageRank).  Default 0.0 (inactive).
    w_geo_urlslash … w_max_urlslash : float
        lC weights for P_urlslash (URL depth).  Default 0.0 (inactive).
    w_geo_urllen … w_max_urllen : float
        lC weights for P_urllen (URL length).  Default 0.0 (inactive).
    n_jobs : int
        Worker threads for parallel query processing (-1 = all CPU cores).
    """

    def __init__(
        self,
        index,
        k: int = 5,
        n_docs: int = 100,
        doc_features: dict | None = None,
        # ── Non-web feature weights (Table 3, non-web setting) ───────────────
        # lQD
        w_geo_qsim:      float = 0.132,
        # lQC
        w_stdv_qsim:     float = 0.143,
        w_max_qsim:      float = 0.121,
        w_min_qsim:      float = 0.088,
        # lC — P_dsim
        w_min_dsim:      float = 0.110,
        w_max_dsim:      float = 0.066,
        w_geo_dsim:      float = 0.055,
        # lC — P_icompress
        w_min_icompress: float = 0.099,
        w_geo_icompress: float = 0.077,
        w_max_icompress: float = 0.044,
        # lC — P_entropy
        w_geo_entropy:   float = 0.033,
        w_min_entropy:   float = 0.022,
        w_max_entropy:   float = 0.011,
        # ── Web feature weights (all default 0 = inactive) ───────────────────
        # lC — P_spam (Waterloo spam score)
        w_geo_spam:      float = 0.0,
        w_min_spam:      float = 0.0,
        w_max_spam:      float = 0.0,
        # lC — P_pr (PageRank)
        w_geo_pr:        float = 0.0,
        w_min_pr:        float = 0.0,
        w_max_pr:        float = 0.0,
        # lC — P_urlslash (URL path depth)
        w_geo_urlslash:  float = 0.0,
        w_min_urlslash:  float = 0.0,
        w_max_urlslash:  float = 0.0,
        # lC — P_urllen (URL length)
        w_geo_urllen:    float = 0.0,
        w_min_urllen:    float = 0.0,
        w_max_urllen:    float = 0.0,
        n_jobs: int = -1,
    ):
        self.index        = index
        self.k            = k
        self.n_docs       = n_docs
        self.doc_features = doc_features  # {docno: {feat: val}}

        self.w_geo_qsim      = w_geo_qsim
        self.w_stdv_qsim     = w_stdv_qsim
        self.w_max_qsim      = w_max_qsim
        self.w_min_qsim      = w_min_qsim
        self.w_min_dsim      = w_min_dsim
        self.w_max_dsim      = w_max_dsim
        self.w_geo_dsim      = w_geo_dsim
        self.w_min_icompress = w_min_icompress
        self.w_geo_icompress = w_geo_icompress
        self.w_max_icompress = w_max_icompress
        self.w_geo_entropy   = w_geo_entropy
        self.w_min_entropy   = w_min_entropy
        self.w_max_entropy   = w_max_entropy

        self.w_geo_spam      = w_geo_spam
        self.w_min_spam      = w_min_spam
        self.w_max_spam      = w_max_spam
        self.w_geo_pr        = w_geo_pr
        self.w_min_pr        = w_min_pr
        self.w_max_pr        = w_max_pr
        self.w_geo_urlslash  = w_geo_urlslash
        self.w_min_urlslash  = w_min_urlslash
        self.w_max_urlslash  = w_max_urlslash
        self.w_geo_urllen    = w_geo_urllen
        self.w_min_urllen    = w_min_urllen
        self.w_max_urllen    = w_max_urllen

        self.n_jobs = n_jobs

        # Precompute whether any web features are active (avoids repeated checks)
        self._use_web = (doc_features is not None) and any(
            v != 0.0 for v in [
                w_geo_spam, w_min_spam, w_max_spam,
                w_geo_pr,   w_min_pr,   w_max_pr,
                w_geo_urlslash, w_min_urlslash, w_max_urlslash,
                w_geo_urllen,   w_min_urllen,   w_max_urllen,
            ]
        )

    def _rerank_query(self, query: str, docs_df: pd.DataFrame) -> pd.DataFrame:
        """Re-rank one query's results using ClustMRF cluster scoring."""
        top  = docs_df.head(self.n_docs).copy().reset_index(drop=True)
        tail = docs_df.iloc[self.n_docs:].copy()
        n    = len(top)
        if n < 2:
            return docs_df

        texts = top["text"].fillna("").tolist()

        # sim(Q, d): softmax-normalise retrieval scores → (0, 1]
        # Works for both BM25 (positive) and DirichletLM (log-prob, negative).
        raw    = np.array(top["score"].tolist(), dtype=float)
        sim_qd = np.exp(raw - raw.max())   # best doc gets 1.0

        # Stemmed TF-IDF cosine similarity matrix (k-NN + P_dsim)
        doc_tokens = [tokenize_and_stem(t) for t in texts]
        k = min(self.k, n)

        vect = TfidfVectorizer(max_features=10_000, sublinear_tf=True,
                               analyzer=lambda d: d)
        try:
            X = vect.fit_transform(doc_tokens)
        except ValueError:
            return docs_df

        Xa = X.toarray() if issparse(X) else np.array(X)
        norms = np.linalg.norm(Xa, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        cos_sim = (Xa / norms) @ (Xa / norms).T   # (n, n) ∈ [-1, 1]

        # Pre-compute per-doc lC measures (called once, reused across clusters)
        entropies   = np.array([_entropy(toks) for toks in doc_tokens])
        icompresses = np.array([_icompress(t)  for t in texts])

        # Web features: look up per-docno values for docs in this top-n set
        if self._use_web:
            docnos = top["docno"].tolist()
            df_map = self.doc_features  # shorthand
            spam_arr     = np.array([df_map.get(d, {}).get("spam",     0.5) for d in docnos])
            pr_arr       = np.array([df_map.get(d, {}).get("pr",       0.0) for d in docnos])
            urlslash_arr = np.array([df_map.get(d, {}).get("urlslash", 0.0) for d in docnos])
            urllen_arr   = np.array([df_map.get(d, {}).get("urllen",   0.0) for d in docnos])

        # Score each cluster C(d_i) and record its member indices
        cluster_nn     = []
        cluster_scores = np.zeros(n)

        for i in range(n):
            nn = np.argsort(cos_sim[i])[::-1][:k]
            cluster_nn.append(nn)
            ki = len(nn)

            # ── lQD: geo-qsim ────────────────────────────────────────────────
            geo_qsim = float(np.log(sim_qd[nn] + EPS).mean())

            # ── lQC: min / max / stdv of sim(Q, d) ───────────────────────────
            sims_nn   = sim_qd[nn]
            min_qsim  = math.log(float(sims_nn.min())  + EPS)
            max_qsim  = math.log(float(sims_nn.max())  + EPS)
            std_nn    = float(sims_nn.std()) if ki > 1 else 0.0
            stdv_qsim = math.log(std_nn + EPS)

            # ── lC: P_dsim — average cosine sim to all cluster members ───────
            pdsim = np.array([float(cos_sim[j, nn].mean()) for j in nn])
            geo_dsim = float(np.log(pdsim + EPS).mean())
            min_dsim = math.log(float(pdsim.min()) + EPS)
            max_dsim = math.log(float(pdsim.max()) + EPS)

            # ── lC: P_entropy ─────────────────────────────────────────────────
            H = entropies[nn]
            geo_entropy = float(np.log(H + EPS).mean())
            min_entropy = math.log(float(H.min()) + EPS)
            max_entropy = math.log(float(H.max()) + EPS)

            # ── lC: P_icompress ───────────────────────────────────────────────
            IC = icompresses[nn]
            geo_icompress = float(np.log(IC + EPS).mean())
            min_icompress = math.log(float(IC.min()) + EPS)
            max_icompress = math.log(float(IC.max()) + EPS)

            score = (
                self.w_geo_qsim      * geo_qsim      +
                self.w_stdv_qsim     * stdv_qsim     +
                self.w_max_qsim      * max_qsim      +
                self.w_min_qsim      * min_qsim      +
                self.w_min_dsim      * min_dsim      +
                self.w_max_dsim      * max_dsim      +
                self.w_geo_dsim      * geo_dsim      +
                self.w_min_icompress * min_icompress +
                self.w_geo_icompress * geo_icompress +
                self.w_max_icompress * max_icompress +
                self.w_geo_entropy   * geo_entropy   +
                self.w_min_entropy   * min_entropy   +
                self.w_max_entropy   * max_entropy
            )

            # ── Web lC features (active only when doc_features provided) ─────
            if self._use_web:
                # P_spam
                SP = spam_arr[nn]
                geo_spam = float(np.log(SP + EPS).mean())
                min_spam = math.log(float(SP.min()) + EPS)
                max_spam = math.log(float(SP.max()) + EPS)

                # P_pr (raw PageRank; log handles wide dynamic range)
                PR = pr_arr[nn]
                geo_pr = float(np.log(PR + EPS).mean())
                min_pr = math.log(float(PR.min()) + EPS)
                max_pr = math.log(float(PR.max()) + EPS)

                # P_urlslash
                US = urlslash_arr[nn]
                geo_urlslash = float(np.log(US + EPS).mean())
                min_urlslash = math.log(float(US.min()) + EPS)
                max_urlslash = math.log(float(US.max()) + EPS)

                # P_urllen
                UL = urllen_arr[nn]
                geo_urllen = float(np.log(UL + EPS).mean())
                min_urllen = math.log(float(UL.min()) + EPS)
                max_urllen = math.log(float(UL.max()) + EPS)

                score += (
                    self.w_geo_spam     * geo_spam     +
                    self.w_min_spam     * min_spam     +
                    self.w_max_spam     * max_spam     +
                    self.w_geo_pr       * geo_pr       +
                    self.w_min_pr       * min_pr       +
                    self.w_max_pr       * max_pr       +
                    self.w_geo_urlslash * geo_urlslash +
                    self.w_min_urlslash * min_urlslash +
                    self.w_max_urlslash * max_urlslash +
                    self.w_geo_urllen   * geo_urllen   +
                    self.w_min_urllen   * min_urllen   +
                    self.w_max_urllen   * max_urllen
                )

            cluster_scores[i] = score

        # ── Cluster → document unrolling (paper §4.1) ────────────────────────
        # Iterate clusters best→worst.  For each cluster, append its docs
        # sorted by sim(Q, d) descending, skipping already-placed docs.
        sorted_centers = np.argsort(cluster_scores)[::-1]
        seen   = set()
        result = []
        for ci in sorted_centers:
            nn = cluster_nn[ci]
            for j in nn[np.argsort(sim_qd[nn])[::-1]]:
                if j not in seen:
                    result.append(int(j))
                    seen.add(j)
        for j in range(n):          # catch any doc not covered (edge case)
            if j not in seen:
                result.append(j)

        # Assign scores: rank-0 doc gets score=n, rank-(n-1) gets score=1
        new_scores = np.zeros(n)
        for rank, idx in enumerate(result):
            new_scores[idx] = n - rank

        top = top.copy()
        top["score"] = new_scores
        top = top.sort_values("score", ascending=False).reset_index(drop=True)
        top["rank"] = range(1, n + 1)

        if not tail.empty:
            tail = tail.copy()
            tail["score"] = [float(new_scores.min()) - len(tail) + i
                             for i in range(len(tail))]
            tail["rank"] = range(n + 1, n + len(tail) + 1)
            return pd.concat([top, tail], ignore_index=True)
        return top

    def transform(self, topics_and_res: pd.DataFrame) -> pd.DataFrame:
        """PyTerrier-compatible transform: re-rank all queries in parallel."""
        groups = [
            (qid, grp.reset_index(drop=True))
            for qid, grp in topics_and_res.groupby("qid", sort=False)
        ]
        if not groups:
            return topics_and_res

        # No Terrier index calls in workers → pure threading is safe
        n_cpu     = os.cpu_count() or 4
        n_workers = min(n_cpu if self.n_jobs == -1 else max(1, self.n_jobs),
                        len(groups))

        def _process(args):
            qid, grp = args
            reranked = self._rerank_query(str(grp["query"].iloc[0]), grp)
            reranked["qid"] = qid
            return reranked

        if n_workers == 1:
            results = [_process(a) for a in groups]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_process, groups))

        out = pd.concat(results, ignore_index=True)
        out["rank"] = (
            out.groupby("qid")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        return out

    def __rrshift__(self, other):
        if _HAS_PT:
            return pt.Transformer.from_callable(
                lambda df: self.transform(other.transform(df))
            )
        raise RuntimeError("PyTerrier not available")
