#!/usr/bin/env bash
set -euo pipefail

# Prepare local cache for TREC-RAG 2024 style experiments on MS MARCO v2.1.
# - Downloads DL'23 topics + v2.1 qrels (small files).
# - Optionally links/downloads the large v2.1 corpus tar.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/trec-rag-2024"
QRELS_DIR="${DATA_DIR}/qrels"
TOPICS_DIR="${DATA_DIR}/topics"
CORPUS_DIR="${DATA_DIR}/corpus"

mkdir -p "${QRELS_DIR}" "${TOPICS_DIR}" "${CORPUS_DIR}"

QRELS_URL="https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.dl23-doc-msmarco-v2.1.txt"
TOPICS_URL="https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.dl23.txt"
CORPUS_URL="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc.tar"

QRELS_PATH="${QRELS_DIR}/qrels.dl23-doc-msmarco-v2.1.txt"
TOPICS_PATH="${TOPICS_DIR}/topics.dl23.txt"
CORPUS_PATH="${CORPUS_DIR}/msmarco_v2.1_doc.tar"

echo "[1/3] Downloading qrels/topics..."
curl -fL "${QRELS_URL}" -o "${QRELS_PATH}"
curl -fL "${TOPICS_URL}" -o "${TOPICS_PATH}"

if [[ -s "${CORPUS_PATH}" ]]; then
  echo "[2/3] Corpus tar already present: ${CORPUS_PATH}"
else
  echo "[2/3] Corpus tar not found."
  echo "      Option A: symlink an existing file to ${CORPUS_PATH}"
  echo "      Option B: download now (~28GB)"
  read -r -p "Download corpus now? [y/N] " REPLY
  if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
    curl -fL "${CORPUS_URL}" -o "${CORPUS_PATH}"
  else
    echo "Skipped corpus download."
  fi
fi

echo "[3/3] Done. Files:"
ls -lh "${QRELS_PATH}" "${TOPICS_PATH}" || true
[[ -e "${CORPUS_PATH}" ]] && ls -lh "${CORPUS_PATH}" || true

echo
cat <<'EOF'
Next step (standard strong baseline):
  1) Build a Lucene BM25 index over the v2.1 corpus with Anserini/Pyserini.
  2) Retrieve top-1000 with BM25.
  3) Rerank top-100 using a cross-encoder (e.g., bge-reranker-v2-m3).

This two-stage setup is the most common modern baseline family in TREC-style IR.
EOF
