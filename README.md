# IR-Experimentations

## Quick Start: MSMARCO v2 sample retrieval

This repo now includes a runnable sample workflow that:

- Downloads data via `ir_datasets`.
- Runs lexical retrieval with BM25 on a sampled corpus.
- Reports `MAP`, `nDCG@10`, and `MRR@10` using `ir_measures`.

Run:

```bash
/Users/asimk/Code/IR-Experimentations/.venv/bin/python main.py --run-msmarco-sample
```

### Dataset defaults

- Default dataset: `msmarco-document-v2/trec-dl-2022/judged`
- Override with `IR_DATASET`, for example:

```bash
IR_DATASET=msmarco-document-v2/trec-dl-2022/judged /Users/asimk/Code/IR-Experimentations/.venv/bin/python main.py --run-msmarco-sample
```

Note: in current `ir_datasets` metadata, MSMARCO v2 TREC-DL `2024`/`2025` splits are not listed. Available MSMARCO v2 TREC-DL IDs currently include `2023` (no qrels) and judged splits through `2022`.

## TREC-RAG 2024 (MS MARCO v2.1 + DL-2023 qrels)

You can prepare the official small evaluation artifacts immediately:

```bash
./scripts/prepare_trec_rag_v21.sh
```

This downloads:

- `data/trec-rag-2024/qrels/qrels.dl23-doc-msmarco-v2.1.txt`
- `data/trec-rag-2024/topics/topics.dl23.txt`

The full corpus (`msmarco_v2.1_doc.tar`, ~28GB) is optional in that script prompt,
but required for local indexing and retrieval over the full collection.

### Standard strong setup

For a common modern TREC-style baseline family:

1. Build a Lucene BM25 index over the v2.1 corpus (Anserini/Pyserini).
2. Retrieve top-1000 with BM25.
3. Rerank top-100 with a cross-encoder (e.g., `bge-reranker-v2-m3` or MonoT5).

This two-stage sparse+rerank pipeline is a standard high-performing approach.