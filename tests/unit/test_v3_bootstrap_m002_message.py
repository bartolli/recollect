"""m002's refusal names an env var the config honors.

The message is the only instruction the operator gets when NULL user_id
rows block the migration; a variable name the config ignores leaves the
server down after the operator follows it.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest
from recollect.bootstrap.migrations.m002_user_id_backfill import (
    _M002UserIdBackfill,
)
from recollect.config import MemoryConfig
from recollect.exceptions import BootstrapError


async def test_refusal_names_an_env_var_the_config_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = AsyncMock()
    conn.fetchval.return_value = 3
    migration = _M002UserIdBackfill(get_default_user_id=lambda: "")

    with pytest.raises(BootstrapError) as refusal:
        await migration.up(conn)

    named = re.findall(r"\b([A-Z][A-Z0-9_]+) env\b", str(refusal.value))
    assert named, "the refusal names no env var"
    for name in named:
        monkeypatch.setenv(name, "u1")
        assert MemoryConfig().server_user_id == "u1", name
        monkeypatch.delenv(name)
    conn.execute.assert_not_called()
