"""Unit: MCP read surfaces pass the server user_id to memory.facts.

Reflect/primer scoping regression net for the fast suite; the two-user
end-to-end parity scenario lives in tests/integration/test_v3_user_isolation.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect_mcp.server import AppContext, _generate_primer, reflect


def _ctx(memory: AsyncMock, user_id: str) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id=user_id
    )
    return ctx


@pytest.fixture()
def memory() -> AsyncMock:
    mem = AsyncMock()
    mem.facts.return_value = []
    return mem


class TestReadSurfaceUserScope:
    async def test_reflect_passes_user_id(self, memory: AsyncMock) -> None:
        await reflect(_ctx(memory, "alex"))
        assert memory.facts.call_count >= 1
        for call in memory.facts.call_args_list:
            assert call.kwargs["user_id"] == "alex"

    async def test_generate_primer_passes_user_id(self, memory: AsyncMock) -> None:
        app = AppContext(memory=memory, worker=MagicMock(), user_id="alex")
        await _generate_primer(app)
        assert memory.facts.call_args.kwargs["user_id"] == "alex"

    async def test_empty_user_id_maps_to_none(self, memory: AsyncMock) -> None:
        app = AppContext(memory=memory, worker=MagicMock(), user_id="")
        await _generate_primer(app)
        assert memory.facts.call_args.kwargs["user_id"] is None
