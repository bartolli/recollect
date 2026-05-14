"""Schema bootstrap: migration registry, runner, preflight invariants.

Public surface: `apply(pool, registry, *, strict)` for one-shot bootstrap
during `CognitiveMemory.connect()` and the operator CLI.
"""

from __future__ import annotations

from recollect.bootstrap.migration import (
    Migration,
    MigrationRegistry,
)

__all__ = [
    "Migration",
    "MigrationRegistry",
]
