"""Seed-loader tests: trace ingest + group restoration shape."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli.corpus import SeedGroup, SeedTrace
from probe_cli.situational_runner import ingest_seed_traces, restore_seed_groups
from recollect.exceptions import MemorySDKError
from recollect.models import MemoryTrace


def _seed_trace(corpus_id: str, group_id: str | None = None) -> SeedTrace:
    return SeedTrace(id=corpus_id, text=f"text-{corpus_id}", seed_group_id=group_id)


def _seed_group(group_id: str, members: list[str]) -> SeedGroup:
    return SeedGroup(
        group_id=group_id,
        person_ref="household",
        situation="overflow",
        implications=["a", "b"],
        significance=0.6,
        member_trace_ids=members,
    )


def _memory_mock(experience_results: list[MemoryTrace] | Exception) -> MagicMock:
    mem = MagicMock()
    if isinstance(experience_results, Exception):
        mem.experience = AsyncMock(side_effect=experience_results)
    else:
        mem.experience = AsyncMock(side_effect=experience_results)
    mem.storage = MagicMock()
    mem.storage.recall_tokens.create_token = AsyncMock()
    mem.storage.recall_tokens.stamp_traces = AsyncMock()
    return mem


class TestIngestSeedTraces:
    async def test_builds_corpus_id_to_trace_id_map(self) -> None:
        traces = [_seed_trace("c1.1", "G1"), _seed_trace("c1.u1")]
        results = [
            MemoryTrace(id=f"uuid-{i}", content=t.text) for i, t in enumerate(traces)
        ]
        mem = _memory_mock(results)
        out = await ingest_seed_traces(mem, traces, session_id="s", user_id="u")
        assert out == {"c1.1": "uuid-0", "c1.u1": "uuid-1"}
        assert mem.experience.await_count == 2

    async def test_skips_failed_ingest(self) -> None:
        traces = [_seed_trace("c1.1"), _seed_trace("c1.2")]
        mem = MagicMock()
        mem.experience = AsyncMock(
            side_effect=[
                MemorySDKError("boom"),
                MemoryTrace(id="uuid-2", content="t2"),
            ]
        )
        out = await ingest_seed_traces(mem, traces, session_id="s", user_id="u")
        assert out == {"c1.2": "uuid-2"}


class TestRestoreSeedGroups:
    async def test_writes_token_with_assembled_label(self) -> None:
        groups = [_seed_group("G1", ["c1.1", "c1.2"])]
        seed_map = {"c1.1": "uuid-1", "c1.2": "uuid-2"}
        mem = _memory_mock([])
        token_map = await restore_seed_groups(mem, groups, seed_map)
        assert "G1" in token_map
        token = mem.storage.recall_tokens.create_token.await_args[0][0]
        assert token.label == "household | overflow | a, b"
        assert token.significance == pytest.approx(0.6)
        assert token.strength == 1.0
        assert token.status == "active"

    async def test_stamps_resolved_member_trace_ids(self) -> None:
        groups = [_seed_group("G1", ["c1.1", "c1.2"])]
        seed_map = {"c1.1": "uuid-1", "c1.2": "uuid-2"}
        mem = _memory_mock([])
        await restore_seed_groups(mem, groups, seed_map)
        token_id, trace_ids = mem.storage.recall_tokens.stamp_traces.await_args[0]
        assert token_id  # uuid string
        assert sorted(trace_ids) == ["uuid-1", "uuid-2"]

    async def test_skips_stamp_when_no_members_resolve(self) -> None:
        groups = [_seed_group("G1", ["missing"])]
        mem = _memory_mock([])
        token_map = await restore_seed_groups(mem, groups, {})
        assert token_map == {}
        mem.storage.recall_tokens.stamp_traces.assert_not_awaited()
        # Token still created — restore_seed_groups creates first, stamps second.
        mem.storage.recall_tokens.create_token.assert_awaited_once()

    async def test_archived_status_propagates(self) -> None:
        groups = [
            SeedGroup(
                group_id="G1",
                person_ref="x",
                situation="y",
                implications=["z"],
                significance=0.5,
                status="archived",
                member_trace_ids=["c1.1"],
            )
        ]
        mem = _memory_mock([])
        await restore_seed_groups(mem, groups, {"c1.1": "uuid-1"})
        token = mem.storage.recall_tokens.create_token.await_args[0][0]
        assert token.status == "archived"
