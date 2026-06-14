"""Orchestrator for baseline and experimental IR pipelines."""

from __future__ import annotations

import argparse
import logging

from config.settings import (
    Settings,
    apply_java_environment,
    configure_logging,
    initialize_pyterrier,
)
from src.workflows.msmarco_sample import RunConfig, run_sample_workflow


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR experiment orchestrator")
    parser.add_argument(
        "--run-msmarco-sample",
        action="store_true",
        help="Run MSMARCO v2 sampled BM25 retrieval + evaluation.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    apply_java_environment(settings)

    if args.run_msmarco_sample:
        run_sample_workflow(
            RunConfig(
                dataset_id=settings.dataset_id,
            )
        )
        return

    initialize_pyterrier(settings)

    logging.getLogger("ir_experiment").info(
        "Initialized IR scaffolding for dataset=%s model=%s",
        settings.dataset_id,
        settings.hf_model_name,
    )


if __name__ == "__main__":
    main()

