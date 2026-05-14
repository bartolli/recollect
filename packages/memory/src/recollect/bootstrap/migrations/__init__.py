"""Migration definitions registered into the default registry.

Each module exports a single Migration instance. The default registry is
assembled in `default_registry()` below — order matches filesystem-name
order (m001 -> m002 -> m003 -> ...). Adding a migration: create
`mNNN_<name>.py` exporting a Migration, then register it here.
"""

from __future__ import annotations

from recollect.bootstrap.migration import MigrationRegistry
from recollect.bootstrap.migrations.m001_initial import m001_initial
from recollect.bootstrap.migrations.m002_user_id_backfill import (
    m002_user_id_backfill,
)
from recollect.bootstrap.migrations.m003_embedding_contract import (
    m003_embedding_contract,
)


def default_registry() -> MigrationRegistry:
    registry = MigrationRegistry()
    registry.register(m001_initial)
    registry.register(m002_user_id_backfill)
    registry.register(m003_embedding_contract)
    return registry


__all__ = [
    "default_registry",
    "m001_initial",
    "m002_user_id_backfill",
    "m003_embedding_contract",
]
