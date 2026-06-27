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

       Non-web document measures (all 19 features per paper §2.1):
         P_dsim(d)      = (1/|C|) Σ_{di∈C} cos_sim(d, di)  [cluster coherence]
         P_entropy(d)   = −Σ_w p(w|d) log p(w|d)            [content breadth]
         P_icompress(d) = |d|_bytes / |gzip(d)|_bytes        [content breadth]
         P_sw1(d)       = stopwords / non-stopwords ratio     [content breadth]
         P_sw2(d)       = fraction of INQUERY stoplist in doc [content breadth]

       Web-specific document measures (ClueWeb09 collections):
         P_spam(d)    = Waterloo spam score ∈ [0,1] (1 = definitely not spam)
         P_pr(d)      = PageRank score (raw; log-transformed internally)
         P_urlslash(d)= number of slashes in the URL path (URL depth)
         P_urllen(d)  = URL character length

sim(Q, d)  = initial retrieval score, softmax-normalised to (0, 1].
cos_sim    = cosine similarity from stemmed TF-IDF vectors.
log uses add-ε (= 1e-10) smoothing before application (paper footnote 1).

Non-web default weights follow the feature-importance order from Table 3
(non-web setting) of Raiber & Kurland 2013, using a proportional scheme:
  w_rank = (20 - rank) / 190   (19 features, ranks 1–19, sums to 1.0)
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

# Ordered list of the 19 non-web feature names.  Index i maps to weight w_<name>
# and to column i of the feature matrix returned by transform_features().
FEATURE_NAMES: list[str] = [
    'geo_qsim',                                    # lQD  (0)
    'stdv_qsim', 'max_qsim', 'min_qsim',          # lQC  (1-3)
    'min_dsim',  'max_dsim', 'geo_dsim',           # lC   (4-6)
    'min_icompress', 'geo_icompress', 'max_icompress',  # lC (7-9)
    'geo_entropy',   'min_entropy',   'max_entropy',    # lC (10-12)
    'max_sw2',       'min_sw2',       'geo_sw2',        # lC (13-15)
    'max_sw1',       'min_sw1',       'geo_sw1',        # lC (16-18)
]

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


# INQUERY stopword list (418 words, UMass INQUERY system, Callan et al.)
_INQUERY_STOPS = frozenset([
    "a", "about", "above", "according", "across", "after", "afterwards", "again",
    "against", "albeit", "all", "almost", "alone", "along", "already", "also",
    "although", "always", "am", "among", "amongst", "an", "and", "another",
    "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anywhere",
    "apart", "are", "around", "as", "at", "av", "be", "became", "because",
    "become", "becomes", "becoming", "been", "before", "beforehand", "behind",
    "being", "below", "beside", "besides", "between", "beyond", "both", "but",
    "by", "can", "cannot", "canst", "certain", "cf", "choose", "contrariwise",
    "cos", "could", "cu", "day", "do", "does", "doesnt", "doing", "dost",
    "doth", "double", "down", "dual", "during", "each", "either", "else",
    "elsewhere", "enough", "et", "etc", "even", "ever", "every", "everybody",
    "everyone", "everything", "everywhere", "except", "excepted", "excepting",
    "exception", "exclude", "excluding", "exclusive", "fairly", "far", "farther",
    "final", "first", "for", "formerly", "forth", "from", "front", "further",
    "furthermore", "get", "go", "had", "halves", "hardly", "has", "hast",
    "hath", "have", "he", "hence", "henceforth", "her", "here", "hereabouts",
    "hereafter", "hereby", "herein", "hereinafter", "hereof", "hereon", "hereto",
    "hereupon", "hers", "herself", "him", "himself", "his", "hither", "hitherto",
    "how", "however", "howsoever", "i", "ie", "if", "in", "inasmuch", "inc",
    "indeed", "indoors", "inside", "insomuch", "instead", "into", "inward",
    "inwards", "is", "it", "its", "itself", "just", "kind", "kg", "km", "last",
    "latter", "latterly", "less", "lest", "let", "like", "likely", "likewise",
    "little", "ltd", "many", "may", "maybe", "me", "meantime", "meanwhile",
    "might", "more", "moreover", "most", "mostly", "much", "must", "my",
    "myself", "namely", "need", "neither", "never", "nevertheless", "next",
    "no", "nobody", "none", "nonetheless", "noone", "nor", "not", "nothing",
    "notwithstanding", "now", "nowadays", "nowhere", "of", "off", "often",
    "ok", "on", "once", "only", "onto", "or", "other", "others", "otherwise",
    "ought", "our", "ourselves", "out", "outside", "over", "own", "per",
    "perhaps", "plenty", "provide", "quite", "rather", "really", "round",
    "said", "same", "save", "self", "several", "shall", "she", "should",
    "since", "so", "some", "somebody", "somehow", "someone", "something",
    "sometime", "sometimes", "somewhere", "still", "such", "than", "that",
    "the", "their", "them", "themselves", "then", "thence", "thenceforth",
    "there", "thereabout", "thereafter", "thereby", "therefore", "therein",
    "thereof", "thereon", "thereto", "thereupon", "these", "they", "this",
    "those", "though", "through", "throughout", "thru", "thus", "till", "to",
    "together", "too", "toward", "towards", "truly", "twice", "under", "until",
    "unless", "unlike", "unlikely", "up", "upon", "us", "very", "via", "vs",
    "was", "we", "well", "were", "what", "whatever", "when", "whence",
    "whenever", "where", "whereabouts", "whereas", "wherefore", "wherein",
    "whereof", "whereon", "whereto", "wherever", "whether", "which", "while",
    "whilst", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "with", "within", "without", "worse", "worst", "would", "yet", "you",
    "your", "yourself", "yourselves",
])

_N_INQUERY_STOPS = len(_INQUERY_STOPS)


def _sw1(tokens: list[str]) -> float:
    """P_sw1: ratio of stopwords to non-stopwords (content breadth measure)."""
    if not tokens:
        return 0.0
    n_sw = sum(1 for t in tokens if t in _INQUERY_STOPS)
    n_non = len(tokens) - n_sw
    return float(n_sw) / float(n_non) if n_non > 0 else float(n_sw)


def _sw2(tokens: list[str]) -> float:
    """P_sw2: fraction of INQUERY stoplist that appears in the document."""
    if not tokens:
        return 0.0
    vocab = frozenset(tokens)
    return sum(1 for sw in _INQUERY_STOPS if sw in vocab) / _N_INQUERY_STOPS


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
    w_geo_qsim … w_geo_sw1 : float
        Feature weights for the 19 non-web features (all lQD, lQC, lC cliques).
        Defaults follow the Table 3 importance ranking (non-ClueWeb) as
        w = (20 − rank) / 190 so all 19 weights sum to 1.0.
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
        n_docs: int = 50,
        doc_features: dict | None = None,
        # ── Non-web feature weights: proportional to Table 3 importance ranks ─
        # w_rank = (20 - rank) / 190  (19 features, sums to 1.0)
        # Importance order (non-ClueWeb): stdv-qsim(1), max-sw2(2), geo-qsim(3),
        #   min-sw2(4), max-sw1(5), max-qsim(6), min-dsim(7), geo-sw2(8),
        #   min-icompress(9), min-qsim(10), min-sw1(11), geo-icompress(12),
        #   max-dsim(13), geo-dsim(14), max-icompress(15), geo-entropy(16),
        #   min-entropy(17), geo-sw1(18), max-entropy(19)
        # lQD
        w_geo_qsim:      float = 0.0895,   # rank 3:  17/190
        # lQC
        w_stdv_qsim:     float = 0.1000,   # rank 1:  19/190
        w_max_qsim:      float = 0.0737,   # rank 6:  14/190
        w_min_qsim:      float = 0.0526,   # rank 10: 10/190
        # lC — P_dsim
        w_min_dsim:      float = 0.0684,   # rank 7:  13/190
        w_max_dsim:      float = 0.0368,   # rank 13:  7/190
        w_geo_dsim:      float = 0.0316,   # rank 14:  6/190
        # lC — P_icompress
        w_min_icompress: float = 0.0579,   # rank 9:  11/190
        w_geo_icompress: float = 0.0421,   # rank 12:  8/190
        w_max_icompress: float = 0.0263,   # rank 15:  5/190
        # lC — P_entropy
        w_geo_entropy:   float = 0.0211,   # rank 16:  4/190
        w_min_entropy:   float = 0.0158,   # rank 17:  3/190
        w_max_entropy:   float = 0.0053,   # rank 19:  1/190
        # lC — P_sw2 (stopword list coverage — ranks 2, 4, 8)
        w_max_sw2:       float = 0.0947,   # rank 2:  18/190
        w_min_sw2:       float = 0.0842,   # rank 4:  16/190
        w_geo_sw2:       float = 0.0632,   # rank 8:  12/190
        # lC — P_sw1 (stopword ratio — ranks 5, 11, 18)
        w_max_sw1:       float = 0.0789,   # rank 5:  15/190
        w_min_sw1:       float = 0.0474,   # rank 11:  9/190
        w_geo_sw1:       float = 0.0105,   # rank 18:  2/190
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
        self.w_max_sw2       = w_max_sw2
        self.w_min_sw2       = w_min_sw2
        self.w_geo_sw2       = w_geo_sw2
        self.w_max_sw1       = w_max_sw1
        self.w_min_sw1       = w_min_sw1
        self.w_geo_sw1       = w_geo_sw1

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

    @property
    def _weight_vector(self) -> np.ndarray:
        """Return the 19 non-web weights as an array aligned with FEATURE_NAMES."""
        return np.array([getattr(self, f'w_{name}') for name in FEATURE_NAMES])

    def _extract_features_for_query(self, docs_df: pd.DataFrame) -> dict | None:
        """Compute the 19 non-web ClustMRF features for each document in the
        initial top-n.  Returns a dict with everything needed to score and
        unroll the clusters, or None if the query has too few documents."""
        top  = docs_df.head(self.n_docs).copy().reset_index(drop=True)
        tail = docs_df.iloc[self.n_docs:].copy()
        n    = len(top)
        if n < 2:
            return None

        texts = top["text"].fillna("").tolist()

        # sim(Q, d): softmax-normalise retrieval scores → (0, 1]
        raw    = np.array(top["score"].tolist(), dtype=float)
        sim_qd = np.exp(raw - raw.max())

        doc_tokens = [tokenize_and_stem(t) for t in texts]
        raw_tokens = [tokenize(t) for t in texts]
        k = min(self.k, n)

        vect = TfidfVectorizer(max_features=10_000, sublinear_tf=True,
                               use_idf=False, analyzer=lambda d: d)
        try:
            X = vect.fit_transform(doc_tokens)
        except ValueError:
            return None

        Xa = X.toarray() if issparse(X) else np.array(X)
        norms = np.linalg.norm(Xa, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        cos_sim = (Xa / norms) @ (Xa / norms).T

        entropies   = np.array([_entropy(toks) for toks in doc_tokens])
        icompresses = np.array([_icompress(t)  for t in texts])
        sw1s        = np.array([_sw1(toks) for toks in raw_tokens])
        sw2s        = np.array([_sw2(toks) for toks in raw_tokens])

        features   = np.zeros((n, len(FEATURE_NAMES)), dtype=float)
        cluster_nn = []

        for i in range(n):
            nn = np.argsort(cos_sim[i])[::-1][:k]
            cluster_nn.append(nn)
            ki = len(nn)

            sims_nn   = sim_qd[nn]
            std_nn    = float(sims_nn.std()) if ki > 1 else 0.0
            pdsim     = np.array([float(cos_sim[j, nn].mean()) for j in nn])
            H, IC, SW1, SW2 = entropies[nn], icompresses[nn], sw1s[nn], sw2s[nn]

            features[i] = [
                float(np.log(sim_qd[nn] + EPS).mean()),          # geo_qsim
                math.log(std_nn + EPS),                           # stdv_qsim
                math.log(float(sims_nn.max()) + EPS),             # max_qsim
                math.log(float(sims_nn.min()) + EPS),             # min_qsim
                math.log(float(pdsim.min()) + EPS),               # min_dsim
                math.log(float(pdsim.max()) + EPS),               # max_dsim
                float(np.log(pdsim + EPS).mean()),                # geo_dsim
                math.log(float(IC.min()) + EPS),                  # min_icompress
                float(np.log(IC + EPS).mean()),                   # geo_icompress
                math.log(float(IC.max()) + EPS),                  # max_icompress
                float(np.log(H + EPS).mean()),                    # geo_entropy
                math.log(float(H.min()) + EPS),                   # min_entropy
                math.log(float(H.max()) + EPS),                   # max_entropy
                math.log(float(SW2.max()) + EPS),                 # max_sw2
                math.log(float(SW2.min()) + EPS),                 # min_sw2
                float(np.log(SW2 + EPS).mean()),                  # geo_sw2
                math.log(float(SW1.max()) + EPS),                 # max_sw1
                math.log(float(SW1.min()) + EPS),                 # min_sw1
                float(np.log(SW1 + EPS).mean()),                  # geo_sw1
            ]

        return {
            'features':   features,    # (n, 19) float64
            'cluster_nn': cluster_nn,  # list of length n, each an int array of size k
            'sim_qd':     sim_qd,      # (n,) softmax-normalised query-doc similarities
            'top':        top,         # DataFrame, first n_docs rows (already copied)
            'tail':       tail,        # DataFrame, rows beyond n_docs
            'n':          n,
        }

    def _apply_weights_and_unroll(self, extracted: dict, weights: np.ndarray) -> pd.DataFrame:
        """Score clusters with `weights`, unroll per §4.1, return re-ranked DataFrame."""
        features   = extracted['features']
        cluster_nn = extracted['cluster_nn']
        sim_qd     = extracted['sim_qd']
        top        = extracted['top'].copy()
        tail       = extracted['tail']
        n          = extracted['n']

        cluster_scores = features @ weights  # (n,)

        # Also add web features if active (weights separate from the 19-vector)
        if self._use_web:
            docnos = top["docno"].tolist()
            df_map = self.doc_features
            spam_arr     = np.array([df_map.get(d, {}).get("spam",     0.5) for d in docnos])
            pr_arr       = np.array([df_map.get(d, {}).get("pr",       0.0) for d in docnos])
            urlslash_arr = np.array([df_map.get(d, {}).get("urlslash", 0.0) for d in docnos])
            urllen_arr   = np.array([df_map.get(d, {}).get("urllen",   0.0) for d in docnos])
            for i, nn in enumerate(cluster_nn):
                SP = spam_arr[nn]; PR = pr_arr[nn]
                US = urlslash_arr[nn]; UL = urllen_arr[nn]
                cluster_scores[i] += (
                    self.w_geo_spam     * float(np.log(SP + EPS).mean())     +
                    self.w_min_spam     * math.log(float(SP.min()) + EPS)    +
                    self.w_max_spam     * math.log(float(SP.max()) + EPS)    +
                    self.w_geo_pr       * float(np.log(PR + EPS).mean())     +
                    self.w_min_pr       * math.log(float(PR.min()) + EPS)    +
                    self.w_max_pr       * math.log(float(PR.max()) + EPS)    +
                    self.w_geo_urlslash * float(np.log(US + EPS).mean())     +
                    self.w_min_urlslash * math.log(float(US.min()) + EPS)    +
                    self.w_max_urlslash * math.log(float(US.max()) + EPS)    +
                    self.w_geo_urllen   * float(np.log(UL + EPS).mean())     +
                    self.w_min_urllen   * math.log(float(UL.min()) + EPS)    +
                    self.w_max_urllen   * math.log(float(UL.max()) + EPS)
                )

        # Cluster → document unrolling (paper §4.1)
        sorted_centers = np.argsort(cluster_scores)[::-1]
        seen, result = set(), []
        for ci in sorted_centers:
            for j in cluster_nn[ci][np.argsort(sim_qd[cluster_nn[ci]])[::-1]]:
                if j not in seen:
                    result.append(int(j)); seen.add(j)
        for j in range(n):
            if j not in seen:
                result.append(j)

        new_scores = np.zeros(n)
        for rank, idx in enumerate(result):
            new_scores[idx] = n - rank

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

    def _rerank_query(self, query: str, docs_df: pd.DataFrame) -> pd.DataFrame:
        extracted = self._extract_features_for_query(docs_df)
        if extracted is None:
            return docs_df
        return self._apply_weights_and_unroll(extracted, self._weight_vector)

    def transform_features(self, topics_and_res: pd.DataFrame) -> pd.DataFrame:
        """Extract per-document feature values for all queries without re-ranking.

        Returns a DataFrame with the original qid/docno/rank/score columns plus
        one column per entry in FEATURE_NAMES, covering the top n_docs documents
        of each query.  Used by ClustMRFSVMRank for cross-validated weight learning.
        """
        groups = [
            (qid, grp.reset_index(drop=True))
            for qid, grp in topics_and_res.groupby("qid", sort=False)
        ]
        rows = []
        for qid, grp in groups:
            extracted = self._extract_features_for_query(grp)
            if extracted is None:
                continue
            top  = extracted['top']
            feats = extracted['features']
            for i in range(extracted['n']):
                r: dict = {
                    'qid':   str(qid),
                    'docno': str(top.at[i, 'docno']),
                    'rank':  int(top.at[i, 'rank']),
                    'score': float(top.at[i, 'score']),
                }
                if 'query' in top.columns:
                    r['query'] = str(top.at[i, 'query'])
                if 'text' in top.columns:
                    r['text'] = str(top.at[i, 'text'])
                for fi, fname in enumerate(FEATURE_NAMES):
                    r[fname] = float(feats[i, fi])
                rows.append(r)
        return pd.DataFrame(rows)

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
