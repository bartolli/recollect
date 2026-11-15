"""Memory -- Human-like memory for AI applications."""

from importlib.metadata import PackageNotFoundError, version

from recollect.core import CognitiveMemory
from recollect.models import (
    Association,
    ConsolidationResult,
    EntityEdge,
    ExperienceResult,
    HealthReport,
    HealthStatus,
    KnownEntity,
    MemoryStats,
    MemoryTrace,
    PersonaFact,
    SurfacedFacts,
    Thought,
    ThoughtWindow,
)
from recollect.storage_context import StorageContext, create_storage_context

try:
    __version__ = version("recollect")
except PackageNotFoundError:  # source-tree import before install
    __version__ = "0.0.0"

__all__ = [
    "Association",
    "CognitiveMemory",
    "ConsolidationResult",
    "EntityEdge",
    "ExperienceResult",
    "HealthReport",
    "HealthStatus",
    "KnownEntity",
    "MemoryStats",
    "MemoryTrace",
    "PersonaFact",
    "StorageContext",
    "SurfacedFacts",
    "Thought",
    "ThoughtWindow",
    "__version__",
    "create_storage_context",
]
