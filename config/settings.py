"""Typed settings and runtime configuration for IR experiments."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass


LOGGER_NAME = "ir_experiment"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    java_home: str | None
    jvm_heap: str
    pyterrier_mem: str
    hf_model_name: str
    dataset_id: str
    log_level: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with sensible defaults."""
        return cls(
            java_home=os.getenv(
                "JAVA_HOME",
                "/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home",
            ),
            jvm_heap=os.getenv("IR_JVM_HEAP", "4G"),
            pyterrier_mem=os.getenv("IR_PYTERRIER_MEM", "4G"),
            hf_model_name=os.getenv(
                "IR_HF_MODEL",
                "Snowflake/snowflake-arctic-embed-l-v1.5",
            ),
            dataset_id=os.getenv(
                "IR_DATASET", "msmarco-document-v2/trec-dl-2022/judged"
            ),
            log_level=getattr(
                logging,
                os.getenv("IR_LOG_LEVEL", "INFO").upper(),
                logging.INFO,
            ),
        )


def configure_logging(level: int) -> None:
    """Configure application logging to stdout."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def apply_java_environment(settings: Settings) -> None:
    """Apply JVM-related environment variables in one isolated place."""
    if settings.java_home:
        os.environ["JAVA_HOME"] = settings.java_home
    os.environ["JVM_OPTS"] = f"-Xms{settings.jvm_heap} -Xmx{settings.jvm_heap}"


def get_torch_device():
    """Return preferred torch device, using Apple Silicon MPS when available."""
    import torch

    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_pyterrier(settings: Settings) -> None:
    """Initialize PyTerrier with configured JVM memory settings."""
    try:
        import pyterrier as pt
    except ImportError:
        logging.getLogger(LOGGER_NAME).warning(
            "PyTerrier is not installed; skipping initialization."
        )
        return

    if not pt.started():
        pt.init(mem=settings.pyterrier_mem)

