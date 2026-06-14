#!/usr/bin/env bash
set -euo pipefail

# Reproducible sparse index build for MS MARCO v2.1 docs.
# Expects extracted shards under:
#   data/trec-rag-2024/corpus/msmarco_v2.1_doc/msmarco_v2.1_doc/*.json.gz

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

INPUT_DIR="${ROOT_DIR}/data/trec-rag-2024/corpus/msmarco_v2.1_doc/msmarco_v2.1_doc"
INDEX_DIR="${ROOT_DIR}/data/indexes/msmarco-v2.1-doc-bm25"
THREADS="${THREADS:-4}"
JVM_HEAP="${JVM_HEAP:-24G}"
RESET_INDEX="${RESET_INDEX:-1}"
VALIDATE_GZIP="${VALIDATE_GZIP:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: Input directory not found: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ "${VALIDATE_GZIP}" == "1" ]]; then
  echo "Validating shard gzip integrity..."
  bad_files=()
  total_files=0
  while IFS= read -r -d '' file; do
    total_files=$((total_files + 1))
    if ! gzip -t "${file}" >/dev/null 2>&1; then
      bad_files+=("${file}")
    fi
  done < <(find "${INPUT_DIR}" -type f -name '*.json.gz' -print0)

  echo "  validated files: ${total_files}"
  if (( ${#bad_files[@]} > 0 )); then
    echo "ERROR: Found corrupted shard(s):" >&2
    for file in "${bad_files[@]}"; do
      echo "  - ${file}" >&2
    done
    echo "Re-extract or replace corrupted shard(s) before indexing." >&2
    exit 1
  fi
fi

mkdir -p "${INDEX_DIR}"

if [[ "${RESET_INDEX}" == "1" ]]; then
  rm -rf "${INDEX_DIR}"
  mkdir -p "${INDEX_DIR}"
fi

echo "Building Lucene index"
echo "  input : ${INPUT_DIR}"
echo "  index : ${INDEX_DIR}"
echo "  threads: ${THREADS}"
echo "  jvm heap: ${JVM_HEAP}"

export JAVA_TOOL_OPTIONS="-Xms4G -Xmx${JVM_HEAP}"

"${PYTHON_BIN}" -m pyserini.index.lucene \
  --collection MsMarcoV2DocCollection \
  --input "${INPUT_DIR}" \
  --index "${INDEX_DIR}" \
  --threads "${THREADS}" \
  --storePositions \
  --storeDocvectors \
  --storeRaw \
  --bm25.accurate

echo "Index build complete: ${INDEX_DIR}"
