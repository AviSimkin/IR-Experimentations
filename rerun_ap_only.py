"""
AP-subcollection-only ClustMRF experiment — matches the paper's 'AP' row in Table 2.

Key differences from the full Disk1+2 notebook:
  - Index: only AP/*.Z files from Disk1 and Disk2
  - n_docs: 50  (paper's D_init = 50)
  - Qrels:  filtered to AP-only document IDs
  - Eval:   MAP@50 (paper's reported cutoff)
"""
import os, sys, re, math, subprocess, tempfile, pathlib, logging, warnings
warnings.filterwarnings('ignore')

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11'
ROOT = pathlib.Path('.').resolve()
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
import ir_measures
from ir_measures import MAP, nDCG, P

import pyterrier as pt
if not pt.java.started():
    pt.java.set_memory_limit(8192)
    pt.java.init()

from src.algorithms.clust_mrf import ClustMRF

# ── Paths ──────────────────────────────────────────────────────────────────
DISK1 = pathlib.Path('/mnt/bi-strg4/pool1-data/DATASETS/i/OLD_STORAGES/storage13/kurland/TREC/Disk1')
DISK2 = pathlib.Path('/mnt/bi-strg4/pool1-data/DATASETS/i/OLD_STORAGES/storage13/kurland/TREC/Disk2')

_QRELS_BASE  = pathlib.Path('/mnt/bi-strg4/pool1-data/DATASETS/i/ds3400/lv_ibm_strg/IBM_STORAGE/USERS_DATA/annabel/qrels_queries')
TOPICS_PATH  = _QRELS_BASE / 'queriesTREC123'
QRELS_PATH   = _QRELS_BASE / 'qrelsTREC123'

_SVMR_BASE   = pathlib.Path('/mnt/bi-strg4/pool1-data/DATASETS/i/ds3400/lv_ibm_strg/IBM_STORAGE/USERS_DATA/liorab/ieir21/supporting_evidence_code')
SVM_LEARN    = _SVMR_BASE / 'svm_rank_learn'
SVM_CLASSIFY = _SVMR_BASE / 'svm_rank_classify'

INDEX_DIR = ROOT / 'data' / 'indexes' / 'trec123_ap_only'
RUNS_DIR  = ROOT / 'data' / 'runs'    / 'trec123_ap_only'
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Corpus (AP only) ────────────────────────────────────────────────────────
_DOC_RE   = re.compile(r'<DOC>(.*?)</DOC>', re.DOTALL)
_DOCNO_RE = re.compile(r'<DOCNO>\s*(.*?)\s*</DOCNO>')
_TEXT_RE  = re.compile(
    r'<(?:TEXT|HEADLINE|TI|DOCTITLE|TITLE|ABSTRACT|HL|H[1-6])>(.*?)</(?:TEXT|HEADLINE|TI|DOCTITLE|TITLE|ABSTRACT|HL|H[1-6])>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r'<[^>]+>')

def _parse_sgml(raw):
    for m in _DOC_RE.finditer(raw):
        body = m.group(1)
        dn   = _DOCNO_RE.search(body)
        if not dn:
            continue
        parts = _TEXT_RE.findall(body)
        text  = ' '.join(p.strip() for p in parts)
        text  = _TAG_RE.sub(' ', text)
        text  = re.sub(r'\s+', ' ', text).strip()
        if text:
            yield {'docno': dn.group(1).strip(), 'text': text}

def _ap_files():
    for disk in [DISK1, DISK2]:
        ap_path = disk / 'AP'
        if not ap_path.exists():
            continue
        for zf in sorted(ap_path.rglob('*.Z')):
            if 'READ' not in zf.name.upper():
                yield zf

def corpus_iter():
    seen    = set()
    z_files = list(_ap_files())
    for zf in tqdm(z_files, desc='Streaming AP .Z files', unit='file'):
        proc = subprocess.Popen(['zcat', str(zf)],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw  = proc.stdout.read().decode('latin-1', errors='replace')
        proc.wait()
        for doc in _parse_sgml(raw):
            if doc['docno'] not in seen:
                seen.add(doc['docno'])
                yield doc

total_z = sum(1 for _ in _ap_files())
print(f'AP .Z files to index: {total_z}')

# ── Build AP-only index ─────────────────────────────────────────────────────
INDEX_DIR.mkdir(parents=True, exist_ok=True)
props_file = INDEX_DIR / 'data.properties'

if not props_file.exists():
    print('Building AP-only index (should take ~5 min)...')
    indexer = pt.IterDictIndexer(
        str(INDEX_DIR),
        overwrite  = True,
        meta       = {'docno': 30, 'text': 131072},
        text_attrs = ['text'],
        blocks     = True,
        tokeniser  = 'EnglishTokeniser',
        stemmer    = 'PorterStemmer',
        stopwords  = 'terrier',
    )
    indexer.index(corpus_iter())
    print('Index built.')
else:
    print(f'AP-only index already exists at {INDEX_DIR}')

index = pt.IndexFactory.of(str(props_file))
stats = index.getCollectionStatistics()
print(f'Documents: {stats.numberOfDocuments:,}   Tokens: {stats.numberOfTokens:,}')

# ── Topics & Qrels (AP-only) ────────────────────────────────────────────────
import xml.etree.ElementTree as ET

tree = ET.parse(str(TOPICS_PATH))
root = tree.getroot()

def _clean_query(text):
    m = re.match(r'#combine\((.*)\)', text.strip())
    return m.group(1).strip() if m else text.strip()

topics_list = []
for q in root.findall('query'):
    num  = q.findtext('number', '').strip()
    text = _clean_query(q.findtext('text', '').strip())
    if num and text:
        topics_list.append({'qid': str(int(num)), 'query': text})
topics_df = pd.DataFrame(topics_list).sort_values('qid', key=lambda s: s.astype(int)).reset_index(drop=True)

# Load ALL qrels then filter to AP-only document IDs
qrels_rows = []
with open(QRELS_PATH) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            qid, _, docid, rel = parts
            qrels_rows.append({'query_id': qid, 'doc_id': docid, 'relevance': int(rel)})
all_qrels_df = pd.DataFrame(qrels_rows)

# AP docs have IDs starting with 'AP'
ap_qrels_df = all_qrels_df[all_qrels_df['doc_id'].str.startswith('AP')].copy()

judged_qids = set(ap_qrels_df['query_id'].unique())
topics_df   = topics_df[topics_df['qid'].isin(judged_qids)].reset_index(drop=True)

qid_docrel = (
    ap_qrels_df[ap_qrels_df['relevance'] > 0]
    .groupby('query_id')
    .apply(lambda g: dict(zip(g['doc_id'], g['relevance'])), include_groups=False)
    .to_dict()
)

rel_per_q = ap_qrels_df[ap_qrels_df['relevance'] > 0].groupby('query_id')['doc_id'].count()
print(f'\nQueries with AP-relevant docs : {len(topics_df)}')
print(f'AP-relevant docs total        : {len(ap_qrels_df[ap_qrels_df["relevance"]>0]):,}')
print(f'Avg relevant AP docs per query: {rel_per_q.mean():.1f}')

# ── SDM retrieval on AP-only index ──────────────────────────────────────────
_sdm_path    = RUNS_DIR / 'sdm.txt'
_sdm_parquet = RUNS_DIR / 'sdm_with_text.parquet'

def _build_sdm():
    sdm_rewrite  = pt.rewrite.SDM()
    dirichlet_br = pt.BatchRetrieve(
        index, wmodel='DirichletLM', num_results=1000,
        metadata=['docno', 'text'],
        controls={'c': 2500},
    )
    import time; t0 = time.time()
    run = (sdm_rewrite >> dirichlet_br).transform(topics_df)
    print(f'SDM done in {time.time()-t0:.1f}s')
    return run

if _sdm_parquet.exists():
    print('Loading cached AP-only SDM run...')
    sdm_run = pd.read_parquet(_sdm_parquet)
else:
    sdm_run = _build_sdm()
    with open(_sdm_path, 'w') as f:
        for qid, grp in sdm_run.sort_values(['qid','rank']).groupby('qid'):
            for rank, row in enumerate(grp.itertuples(), 1):
                f.write(f'{qid} Q0 {row.docno} {rank} {row.score:.6f} SDM_AP\n')
    sdm_run.to_parquet(_sdm_parquet)
    print(f'SDM saved')

print(f'SDM run: {len(sdm_run):,} rows, {sdm_run["qid"].nunique()} queries')

# Baseline MAP@50 vs. paper
sdm_eval = sdm_run.rename(columns={'docno':'doc_id','qid':'query_id'})
sdm_agg  = ir_measures.calc_aggregate([MAP @ 50, MAP], ap_qrels_df, sdm_eval)
print(f'\nSDM MAP@50 (AP-only qrels): {float(sdm_agg[MAP@50]):.4f}  (paper Init = 0.101)')
print(f'SDM MAP    (AP-only qrels): {float(sdm_agg[MAP]):.4f}')

# ── Feature extraction  (n_docs=50, matching paper D_init) ─────────────────
N_DOCS, K_CLUSTER = 50, 5
extractor = ClustMRF(index=None, k=K_CLUSTER, n_docs=N_DOCS)

qid_data: dict = {}
for qid, grp in tqdm(sdm_run.groupby('qid'), desc='Extracting features'):
    feats, cluster_nn, sim_qd, top_df = extractor.extract_cluster_features(
        grp.reset_index(drop=True)
    )
    qid_data[qid] = {'feats': feats, 'cluster_nn': cluster_nn,
                     'sim_qd': sim_qd, 'top_df': top_df}

print(f'Features extracted: {len(qid_data)} queries, '
      f'{sum(d["feats"].shape[0] for d in qid_data.values())} clusters')

# ── SVM^rank helpers (NDCG@k labels) ───────────────────────────────────────
def _cluster_ndcg(nn_docnos, sim_qd_cluster, doc_rels):
    order = np.argsort(sim_qd_cluster)[::-1]
    rels  = [1 if doc_rels.get(nn_docnos[j], 0) > 0 else 0 for j in order]
    n_rel = sum(rels)
    if n_rel == 0: return 0.0
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg

def _write_svmrank_file(fpath, qids, qid_data, qid_docrel, qid_int_map):
    offsets, idx = {}, 0
    with open(fpath, 'w') as f:
        for qid in qids:
            data       = qid_data[qid]
            feats      = data['feats']
            cluster_nn = data['cluster_nn']
            sim_qd     = data['sim_qd']
            top_df     = data['top_df']
            n          = len(feats)
            offsets[qid] = (idx, n)
            q_int    = qid_int_map[qid]
            doc_rels = qid_docrel.get(qid, {})
            for i in range(n):
                nn             = cluster_nn[i]
                nn_docnos      = [top_df.iloc[j]['docno'] for j in nn]
                sim_qd_cluster = sim_qd[nn]
                c_rel          = _cluster_ndcg(nn_docnos, sim_qd_cluster, doc_rels)
                feat_str       = ' '.join(f'{j+1}:{v:.8f}' for j, v in enumerate(feats[i]))
                f.write(f'{c_rel} qid:{q_int} {feat_str}\n')
            idx += n
    return offsets

def _run_svmrank(train_file, test_file, C, tmpdir):
    model_file = tmpdir / f'model_C{C}.dat'
    pred_file  = tmpdir / f'pred_C{C}.dat'
    r = subprocess.run([str(SVM_LEARN), '-c', str(C), '-t', '0',
                        str(train_file), str(model_file)],
                       capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(f'learn C={C}: {r.stderr[-200:]}')
    r = subprocess.run([str(SVM_CLASSIFY), str(test_file), str(model_file), str(pred_file)],
                       capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(f'classify: {r.stderr[-200:]}')
    with open(pred_file) as fh:
        return [float(l.strip()) for l in fh if l.strip()]

def _parse_predictions(raw_preds, qids, offsets):
    return {qid: np.array(raw_preds[s:s+n]) for qid, (s, n) in offsets.items() if qid in qids}

def _scores_to_run(qids, qid_data, qid_scores, full_sdm=None):
    tail_by_qid = {}
    if full_sdm is not None:
        for qid, grp in full_sdm.groupby('qid'):
            n_top = qid_data[qid]['feats'].shape[0]
            tail  = grp.sort_values('rank').iloc[n_top:]
            if not tail.empty:
                tail_by_qid[qid] = tail
    rows = []
    for qid in qids:
        data       = qid_data[qid]
        scores     = np.array(qid_scores[qid])
        cluster_nn = data['cluster_nn']
        sim_qd     = data['sim_qd']
        top_df     = data['top_df']
        n          = len(scores)
        seen, result = set(), []
        for ci in np.argsort(scores)[::-1]:
            for j in cluster_nn[ci][np.argsort(sim_qd[cluster_nn[ci]])[::-1]]:
                if j not in seen:
                    result.append(int(j)); seen.add(j)
        for j in range(n):
            if j not in seen: result.append(j)
        for rank, idx in enumerate(result):
            rows.append({'qid': qid, 'docno': top_df.iloc[idx]['docno'],
                         'rank': rank+1, 'score': float(n - rank)})
        tail = tail_by_qid.get(qid)
        if tail is not None:
            for i, tr in enumerate(tail.itertuples()):
                rows.append({'qid': qid, 'docno': tr.docno,
                             'rank': n+i+1, 'score': -float(i)})
    return pd.DataFrame(rows)

def _eval_map50(run_df):
    ev = run_df.rename(columns={'qid': 'query_id', 'docno': 'doc_id'})
    return float(ir_measures.calc_aggregate([MAP @ 50], ap_qrels_df, ev)[MAP @ 50])

# ── Nested 5×5 CV ──────────────────────────────────────────────────────────
print('\n--- Nested 5×5 CV (AP-only, n_docs=50, NDCG@k labels) ---')
OUTER_K, INNER_K = 5, 5
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

qids        = sorted(qid_data.keys(), key=int)
qid_int_map = {qid: i+1 for i, qid in enumerate(qids)}
outer_kf    = KFold(n_splits=OUTER_K, shuffle=True, random_state=42)
qids_arr    = np.array(qids)

cv_fold_runs, best_C_per_fold = [], []

with tempfile.TemporaryDirectory(prefix='clustmrf_ap_') as tmpdir_str:
    tmpdir = pathlib.Path(tmpdir_str)

    for outer_fold, (tr_idx, te_idx) in enumerate(outer_kf.split(qids_arr)):
        train_qids = qids_arr[tr_idx].tolist()
        test_qids  = qids_arr[te_idx].tolist()
        print(f'\n=== Outer fold {outer_fold+1}/{OUTER_K} '
              f'(train={len(train_qids)}, test={len(test_qids)}) ===')

        inner_kf     = KFold(n_splits=INNER_K, shuffle=True, random_state=0)
        train_arr    = np.array(train_qids)
        c_map50s     = {C: [] for C in C_VALUES}

        for inner_fold, (itr_idx, val_idx) in enumerate(inner_kf.split(train_arr)):
            inner_train_q = train_arr[itr_idx].tolist()
            val_q         = train_arr[val_idx].tolist()

            itr_file = tmpdir / f'o{outer_fold}_i{inner_fold}_train.dat'
            val_file = tmpdir / f'o{outer_fold}_i{inner_fold}_val.dat'
            _write_svmrank_file(itr_file, inner_train_q, qid_data, qid_docrel, qid_int_map)
            val_offsets = _write_svmrank_file(val_file, val_q, qid_data, qid_docrel, qid_int_map)

            for C in C_VALUES:
                try:
                    preds   = _run_svmrank(itr_file, val_file, C, tmpdir)
                    val_sc  = _parse_predictions(preds, val_q, val_offsets)
                    val_run = _scores_to_run(val_q, qid_data, val_sc, full_sdm=sdm_run)
                    c_map50s[C].append(_eval_map50(val_run))
                except Exception as exc:
                    print(f'  inner {inner_fold+1} C={C} failed: {exc}')
                    c_map50s[C].append(0.0)

        best_C = max(C_VALUES, key=lambda C: np.mean(c_map50s[C]))
        best_C_per_fold.append(best_C)
        print('  Inner-fold MAP@50 by C:')
        for C in C_VALUES:
            marker = ' ←' if C == best_C else ''
            print(f'    C={C:<8} MAP@50={np.mean(c_map50s[C]):.4f}{marker}')

        full_train_file = tmpdir / f'o{outer_fold}_fulltrain.dat'
        test_file_path  = tmpdir / f'o{outer_fold}_test.dat'
        _write_svmrank_file(full_train_file, train_qids, qid_data, qid_docrel, qid_int_map)
        test_offsets = _write_svmrank_file(test_file_path, test_qids, qid_data, qid_docrel, qid_int_map)

        preds    = _run_svmrank(full_train_file, test_file_path, best_C, tmpdir)
        test_sc  = _parse_predictions(preds, test_qids, test_offsets)
        fold_run = _scores_to_run(test_qids, qid_data, test_sc, full_sdm=sdm_run)
        print(f'  Test MAP@50 (fold {outer_fold+1}) = {_eval_map50(fold_run):.4f}')
        cv_fold_runs.append(fold_run)

print(f'\nBest C per fold: {best_C_per_fold}')
clustmrf_cv_run = pd.concat(cv_fold_runs, ignore_index=True)

# ── Fixed-weight ClustMRF ───────────────────────────────────────────────────
cm_fixed = ClustMRF(index=None, k=K_CLUSTER, n_docs=N_DOCS)
clustmrf_fixed_run = cm_fixed.transform(sdm_run)

# ── Final results ──────────────────────────────────────────────────────────
MEASURES = [MAP @ 50, MAP @ 1000, P @ 5, nDCG @ 5, nDCG @ 10]

def _agg(run):
    ev = run.rename(columns={'docno':'doc_id','qid':'query_id'})
    return ir_measures.calc_aggregate(MEASURES, ap_qrels_df, ev)

bm25_run_path = pathlib.Path('data/runs/trec123/bm25_with_text.parquet')
if bm25_run_path.exists():
    bm25_run = pd.read_parquet(bm25_run_path)
    bm25_agg = _agg(bm25_run)
else:
    bm25_agg = None

sdm_agg2  = _agg(sdm_run)
fixed_agg = _agg(clustmrf_fixed_run)
cv_agg    = _agg(clustmrf_cv_run)

print('\n=== AP-Only Results (AP-only qrels, n_docs=50) ===')
rows = []
for name, agg in [('SDM', sdm_agg2), ('ClustMRF (fixed)', fixed_agg),
                   ('ClustMRF (CV, NDCG@k)', cv_agg)]:
    row = {'System': name}
    for m in MEASURES:
        row[str(m)] = round(float(agg[m]), 4)
    rows.append(row)

results_df = pd.DataFrame(rows).set_index('System')
print(results_df.to_string())
print()
print('Paper Table 2 (AP row):')
print('  Init (SDM) MAP@50 = 0.101  |  ClustMRF MAP@50 = 0.108')
print()
print('Δ ClustMRF CV − SDM:')
for m in MEASURES:
    delta = float(cv_agg[m]) - float(sdm_agg2[m])
    print(f'  {str(m):<14}: {delta:+.4f}')
