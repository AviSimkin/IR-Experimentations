"""Orchestrator for baseline and experimental IR pipelines."""

from __future__ import annotations

import logging

from config.settings import (
    Settings,
    apply_java_environment,
    configure_logging,
    initialize_pyterrier,
)


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    apply_java_environment(settings)
    initialize_pyterrier(settings)

    logging.getLogger("ir_experiment").info(
        "Initialized IR scaffolding for dataset=%s model=%s",
        settings.dataset_id,
        settings.hf_model_name,
    )


if __name__ == "__main__":
    main()

