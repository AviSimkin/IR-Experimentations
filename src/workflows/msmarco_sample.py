"""Sample MSMARCO v2 workflow: download, retrieve with BM25, and evaluate."""

from __future__ import annotations

import argparse
import logging
import random
import re
from dataclasses import dataclass
from typing import Any

import ir_measures
import numpy as np
from ir_measures import MAP, MRR, nDCG
from rank_bm25 import BM25Okapi

from src.data.stream import load_dataset

LOGGER = logging.getLogger("ir_experiment")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RunConfig:
    dataset_id: str = "msmarco-document-v2/trec-dl-2022/judged"
    query_limit: int = 25
    distractor_docs: int = 4000
    top_k: int = 100
    random_seed: int = 13


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _query_id(query: Any) -> str:
    for attr in ("query_id", "qid", "id"):
        if hasattr(query, attr):
            return str(getattr(query, attr))
    raise ValueError(f"Unsupported query object: {type(query)!r}")


def _query_text(query: Any) -> str:
    for attr in ("text", "query", "title", "description"):
        value = getattr(query, attr, None)
        if value:
            return str(value)
    raise ValueError(f"No text-like field found for query object: {type(query)!r}")


def _doc_id(doc: Any) -> str:
    for attr in ("doc_id", "docno", "id"):
        if hasattr(doc, attr):
            return str(getattr(doc, attr))
    raise ValueError(f"Unsupported document object: {type(doc)!r}")


def _doc_text(doc: Any) -> str:
    title = str(getattr(doc, "title", "") or "")
    for attr in ("text", "body", "contents", "passage"):
        value = getattr(doc, attr, None)
        if value:
            return " ".join(part for part in (title, str(value)) if part).strip()
    if title:
        return title
    raise ValueError(f"No text-like field found for document object: {type(doc)!r}")


def _sample_queries(dataset: Any, query_limit: int, random_seed: int) -> dict[str, str]:
    all_queries = list(dataset.queries_iter())
    if not all_queries:
        raise RuntimeError("Dataset has no queries; cannot run retrieval workflow.")

    random.Random(random_seed).shuffle(all_queries)
    sampled = all_queries[: min(query_limit, len(all_queries))]
    return {_query_id(query): _query_text(query) for query in sampled}


def _sample_qrels(dataset: Any, allowed_qids: set[str]) -> list[dict[str, Any]]:
    qrels: list[dict[str, Any]] = []
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        if qid not in allowed_qids:
            continue
        qrels.append(
            {
                "query_id": qid,
                "doc_id": str(qrel.doc_id),
                "relevance": int(qrel.relevance),
            }
        )
    return qrels


def _build_doc_pool(dataset: Any, qrels: list[dict[str, Any]], distractor_docs: int) -> dict[str, str]:
    relevant_doc_ids = {row["doc_id"] for row in qrels if row["relevance"] > 0}
    docs: dict[str, str] = {}

    docs_store = dataset.docs_store()
    if docs_store is not None:
        for did in relevant_doc_ids:
            doc = docs_store.get(did)
            if doc is None:
                continue
            try:
                docs[did] = _doc_text(doc)
            except ValueError:
                continue

    if len(docs) < len(relevant_doc_ids):
        missing = relevant_doc_ids.difference(docs.keys())
        for doc in dataset.docs_iter():
            did = _doc_id(doc)
            if did not in missing:
                continue
            try:
                docs[did] = _doc_text(doc)
            except ValueError:
                continue
            if len(docs) == len(relevant_doc_ids):
                break

    for doc in dataset.docs_iter():
        if len(docs) >= len(relevant_doc_ids) + distractor_docs:
            break
        did = _doc_id(doc)
        if did in docs:
            continue
        try:
            docs[did] = _doc_text(doc)
        except ValueError:
            continue

    if not docs:
        raise RuntimeError("No documents could be loaded from the dataset.")
    return docs


def _run_bm25(
    queries: dict[str, str],
    docs: dict[str, str],
    top_k: int,
) -> list[dict[str, Any]]:
    doc_ids = list(docs.keys())
    tokenized_docs = [_tokenize(docs[did]) for did in doc_ids]
    bm25 = BM25Okapi(tokenized_docs)

    run_rows: list[dict[str, Any]] = []
    for qid, text in queries.items():
        tokens = _tokenize(text)
        if not tokens:
            continue

        scores = bm25.get_scores(tokens)
        k = min(top_k, len(doc_ids))
        top_idx = np.argsort(scores)[::-1][:k]
        for rank, idx in enumerate(top_idx, start=1):
            run_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_ids[int(idx)],
                    "score": float(scores[int(idx)]),
                }
            )
            if rank >= k:
                break

    return run_rows


def _resolve_dataset_with_qrels(dataset_id: str) -> Any:
    dataset = load_dataset(dataset_id)
    if hasattr(dataset, "qrels_iter"):
        return dataset

    candidates: list[str] = []
    if not dataset_id.endswith("/judged"):
        candidates.append(f"{dataset_id}/judged")

    for candidate in candidates:
        try:
            fallback = load_dataset(candidate)
        except KeyError:
            continue
        if hasattr(fallback, "qrels_iter"):
            LOGGER.info(
                "Dataset %s has no qrels; using %s for evaluation",
                dataset_id,
                candidate,
            )
            return fallback

    raise RuntimeError(
        "Dataset has no qrels and no direct '/judged' variant was found. "
        f"Requested: {dataset_id}. "
        "For ir_datasets MSMARCO v2 TREC-DL, judged splits are currently available "
        "through 2022."
    )


def run_sample_workflow(config: RunConfig) -> dict[str, float]:
    LOGGER.info("Loading dataset %s", config.dataset_id)
    dataset = _resolve_dataset_with_qrels(config.dataset_id)

    queries = _sample_queries(dataset, config.query_limit, config.random_seed)
    qrels = _sample_qrels(dataset, set(queries.keys()))
    docs = _build_doc_pool(dataset, qrels, config.distractor_docs)

    LOGGER.info(
        "Sampled %d queries, %d qrels, and %d documents",
        len(queries),
        len(qrels),
        len(docs),
    )

    run_rows = _run_bm25(queries, docs, config.top_k)
    if not run_rows:
        raise RuntimeError("Retriever produced no results.")

    measures = [MAP, nDCG @ 10, MRR @ 10]
    summary = ir_measures.calc_aggregate(measures, qrels, run_rows)

    for measure in measures:
        value = float(summary[measure])
        LOGGER.info("%s = %.4f", measure, value)

    return {str(measure): float(summary[measure]) for measure in measures}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small BM25 retrieval+evaluation workflow on MSMARCO v2 TREC-DL."
        )
    )
    parser.add_argument(
        "--dataset-id",
        default=RunConfig.dataset_id,
        help="ir_datasets dataset id to use.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=RunConfig.query_limit,
        help="Number of queries to sample.",
    )
    parser.add_argument(
        "--distractor-docs",
        type=int,
        default=RunConfig.distractor_docs,
        help="How many non-relevant docs to add to sampled corpus.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RunConfig.top_k,
        help="Retriever cutoff for each query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RunConfig.random_seed,
        help="Random seed for query sampling.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_sample_workflow(
        RunConfig(
            dataset_id=args.dataset_id,
            query_limit=args.query_limit,
            distractor_docs=args.distractor_docs,
            top_k=args.top_k,
            random_seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
