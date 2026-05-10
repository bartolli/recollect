"""Recollect prompt A/B harness.

Measures extraction-prompt quality on a fixture corpus, isolated per arm.
P3 scope: extraction-only (no DB, no embeddings, no retrieval).
"""

from probe_cli.arm import Arm, load_arm
from probe_cli.corpus import (
    CorpusEntry,
    EvalEntry,
    SeedGroup,
    SeedTrace,
    load_corpus,
    load_eval_corpus,
    load_seed_groups,
    load_seed_traces,
)
from probe_cli.metrics import AggregateMetrics, EntryResult, RunReport
from probe_cli.runner import ArmRunner

__all__ = [
    "AggregateMetrics",
    "Arm",
    "ArmRunner",
    "CorpusEntry",
    "EntryResult",
    "EvalEntry",
    "RunReport",
    "SeedGroup",
    "SeedTrace",
    "load_arm",
    "load_corpus",
    "load_eval_corpus",
    "load_seed_groups",
    "load_seed_traces",
]
