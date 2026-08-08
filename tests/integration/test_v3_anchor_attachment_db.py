"""Integration: opaque chain tail attaches to its group via entity anchor.

Story-7 scenario 1 against real storage: under interleaved arrival with the
semantic gate closed (write_time_threshold above cosine ceiling), the tail's
entity edge carries it into assessment, the scripted assessor groups it with
the bridge, and the one-hop token propagation from the bridge reaches the
tail. LLM and embeddings are boundary-mocked; associations SQL, stamping,
and the hop CTE are real.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import asyncpg
import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import Entity, ExtractionResult, TokenAssessment

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]

_EMB_DIM = 768


def _emb(seed: float) -> list[float]:
    return [seed + i * 0.001 for i in range(_EMB_DIM)]


def _admin_url() -> str:
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path="/postgres"))


def _scratch_url(db_name: str) -> str:
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _extraction(entities: list[str]) -> ExtractionResult:
    return ExtractionResult(
        concepts=["chain"],
        entities=[Entity(name=n) for n in entities],
        significance=0.6,
    )


@pytest.fixture()
async def mem() -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"anchor_att_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    embeddings = AsyncMock()
    embeddings.generate_embedding = AsyncMock(side_effect=lambda *_a, **_k: _emb(0.1))
    embeddings.generate_embeddings_batch = AsyncMock(
        side_effect=lambda texts, **_: [_emb(0.1) for _ in texts]
    )
    embeddings.dimensions = _EMB_DIM
    embeddings.contract = lambda: ("nomic-ai/nomic-embed-text-v1.5", "v0.7")

    extractor = AsyncMock()
    config = MemoryConfig()
    # Cosine ceiling is 1.0: threshold 2.0 closes the semantic gate so
    # admission is attributable to the anchor path alone.
    config._set("recall_tokens.write_time_threshold", 2.0)

    m = CognitiveMemory(embeddings=embeddings, extractor=extractor, config=config)
    await m.connect(_scratch_url(db_name))
    try:
        yield m
    finally:
        await m.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class TestOpaqueTailAttachesViaEntityAnchor:
    async def test_tail_joins_group_and_hop_reaches_it(self, mem: CognitiveMemory):
        extractor = mem._extractor
        assert extractor is not None
        extractor.extract = AsyncMock(
            side_effect=[
                _extraction(["Pat", "auto-injector"]),
                _extraction(["volcano"]),
                _extraction(["volcano"]),
                _extraction(["auto-injector"]),
            ]
        )

        # Anchored ordering is not part of the contract (bulk fetch is
        # unordered); the scripted assessor finds the bridge line by content,
        # as the real LLM would.
        def _assess(messages, *args, **kwargs):
            prompt = messages[1].content
            if "refill" not in prompt:
                return TokenAssessment(action="none")
            for line in prompt.splitlines():
                if (
                    line.strip().startswith(tuple("123456789"))
                    and "epinephrine" in line
                ):
                    idx = int(line.strip().split(".", 1)[0])
                    return TokenAssessment(
                        action="create",
                        linked_indices=[idx],
                        person_ref="Pat",
                        situation="allergy management",
                        implication="refill before expiry",
                        significance=0.8,
                    )
            return TokenAssessment(action="none")

        extractor._provider = AsyncMock()
        extractor._provider.complete_structured = AsyncMock(side_effect=_assess)

        bridge = await mem.experience(
            "Doctor prescribed an epinephrine auto-injector for Pat",
            session_id="s1",
            user_id="u1",
        )
        await mem.experience(
            "Sofi chose a volcano for the science fair",
            session_id="s1",
            user_id="u1",
        )
        await mem.experience(
            "The volcano needs baking soda for the eruption",
            session_id="s1",
            user_id="u1",
        )
        tail = await mem.experience(
            "The auto-injector refill is due; the current one expires March 30",
            session_id="s1",
            user_id="u1",
        )

        create_call = extractor._provider.complete_structured.await_args_list[-1]
        numbered = create_call.args[0][1].content
        assert bridge.content in numbered

        # stamped_trace_ids is the intersection with the queried ids, so
        # membership asserts split: group-on-bridge, group-on-tail, then the
        # hop proving both sit on the same token.
        for seed in (bridge.id, tail.id):
            groups = await mem.storage.recall_tokens.find_groups_for_traces(
                [seed], user_id="u1"
            )
            assert len(groups) == 1
            assert groups[0]["label"] == (
                "Pat | allergy management | refill before expiry"
            )

        activated = await mem.storage.recall_tokens.get_activated_trace_ids(
            [bridge.id], user_id="u1"
        )
        assert tail.id in {row[0] for row in activated}
