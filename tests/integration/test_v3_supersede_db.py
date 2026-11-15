"""Integration: supersede replaces the statement on a real store."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.exceptions import TraceNotFoundError
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import FactCategory, PersonaFact
from recollect_mcp.server import AppContext
from recollect_mcp.server import forget as mcp_forget

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)
USER = "u1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


def _extraction(
    predicate: str,
    obj: str,
    *,
    category: FactCategory,
    context: str,
    tags: list[str],
) -> ExtractionResult:
    relation = Relation(
        source="user",
        relation=predicate,
        target=obj,
        confidence=0.9,
        category=category,
        context=context,
        context_tags=tags,
    )
    return ExtractionResult(
        fact_type="semantic", concepts=[obj], relations=[relation], significance=0.6
    )


@pytest.fixture()
def extractor() -> AsyncMock:
    x = AsyncMock()
    x.extract = AsyncMock(return_value=ExtractionResult())
    return x


@pytest.fixture()
async def mem(extractor: AsyncMock) -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"supersede_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_url("postgres"))
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    cfg = MemoryConfig()
    cfg._set("recall_tokens.enabled", False)
    m = CognitiveMemory(extractor=extractor, config=cfg)
    await m.connect(_url(db_name))
    try:
        yield m
    finally:
        await m.close()
        admin = await asyncpg.connect(_url("postgres"))
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


async def _facts(mem: CognitiveMemory) -> list[PersonaFact]:
    return await mem.storage.facts.get_persona_facts(user_id=USER, limit=None)


async def _fetchval(mem: CognitiveMemory, sql: str, *args: object) -> object:
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(sql, *args)


async def test_supersede_replaces_the_statement(
    mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    extractor.extract.return_value = _extraction(
        "prefers",
        "green tea",
        category="preference",
        context="The user prefers green tea.",
        tags=["tea"],
    )
    a = await mem.experience("I prefer green tea", user_id=USER)
    extractor.extract.return_value = _extraction(
        "prefers",
        "green tea",
        category="preference",
        context="The user prefers green tea, without sugar.",
        tags=["tea"],
    )
    result = await mem.experience_result(
        "I prefer green tea, without sugar", user_id=USER, supersedes=[a.id]
    )

    stored_a = await mem.storage.traces.get_trace(a.id)
    assert stored_a is not None
    assert stored_a.status == "forgotten"
    facts = await _facts(mem)
    assert [f.status for f in facts if f.source_trace_id == a.id] == ["archived"]
    (live,) = [f for f in facts if f.status != "archived"]
    assert live.content == "The user prefers green tea, without sugar."
    assert live.source_trace_id == result.trace.id
    assert live.mention_count == 1


@pytest.mark.parametrize(
    ("predicate", "obj", "category", "kept"),
    [
        ("prefers", "green tea", "preference", "pinned"),
        ("is_allergic_to", "peanuts", "health", "promoted"),
    ],
    ids=["pinned", "health"],
)
async def test_a_retained_row_adopts_the_restatement(
    predicate: str,
    obj: str,
    category: FactCategory,
    kept: str,
    mem: CognitiveMemory,
    extractor: AsyncMock,
) -> None:
    extractor.extract.return_value = _extraction(
        predicate,
        obj,
        category=category,
        context="Said in passing.",
        tags=["old tag"],
    )
    a = await mem.experience(f"user {predicate} {obj}", user_id=USER)
    if kept == "pinned":
        await mem.pin(a.id)
    (row,) = await _facts(mem)
    extractor.extract.return_value = _extraction(
        predicate,
        obj,
        category=category,
        context="Confirmed by the user's doctor.",
        tags=["doctor visit"],
    )
    result = await mem.experience_result(
        f"Confirmed: user {predicate} {obj}", user_id=USER, supersedes=[a.id]
    )

    assert [f.id for f in result.superseded[0].retained_facts] == [row.id]
    (adopted,) = await _facts(mem)
    assert adopted.id == row.id
    assert adopted.status == kept
    assert adopted.content == "Confirmed by the user's doctor."
    assert adopted.source_trace_id == result.trace.id
    assert adopted.embedding is not None
    assert adopted.embedding != row.embedding
    concepts = await _fetchval(
        mem,
        "SELECT array_agg(concept) FROM concept_embeddings "
        "WHERE owner_type = 'fact' AND owner_id = $1",
        row.id,
    )
    assert concepts == ["doctor visit"]


@pytest.mark.parametrize("owner", ["missing", "u2"])
async def test_a_refused_supersede_stores_no_trace(
    owner: str, mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    other = await mem.experience("I prefer green tea", user_id="u2")
    target = other.id if owner == "u2" else str(uuid.uuid4())
    calls = extractor.extract.await_count

    with pytest.raises(TraceNotFoundError):
        await mem.experience_result(
            "I prefer black coffee", user_id=USER, supersedes=[target]
        )

    assert await _fetchval(mem, "SELECT count(*) FROM memory_traces") == 1
    assert extractor.extract.await_count == calls
    stored = await mem.storage.traces.get_trace(other.id)
    assert stored is not None
    assert stored.status == "active"


async def test_forced_forget_by_short_id_removes_a_retained_fact(
    mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    extractor.extract.return_value = _extraction(
        "is_allergic_to",
        "peanuts",
        category="health",
        context="Allergic to peanuts.",
        tags=[],
    )
    a = await mem.experience("I am allergic to peanuts", user_id=USER)
    extractor.extract.return_value = ExtractionResult(significance=0.5)
    result = await mem.experience_result(
        "No allergy after all", user_id=USER, supersedes=[a.id]
    )
    (retained,) = result.superseded[0].retained_facts
    ctx = MagicMock()
    ctx.request_context.lifespan_context = AppContext(
        memory=mem, worker=MagicMock(), user_id=USER
    )
    ctx.headers = None

    repeat = await mcp_forget(a.id[:8], ctx)
    assert "1 facts retained (health); repeat with force=true" in repeat
    reply = await mcp_forget(a.id[:8], ctx, force=True)

    assert "1 derived facts archived." in reply
    (row,) = await _facts(mem)
    assert (row.id, row.status) == (retained.id, "archived")


async def test_a_promoted_fact_hands_its_status_to_the_same_slot(
    mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    extractor.extract.return_value = _extraction(
        "works_at", "Acme", category="identity", context="Works at Acme.", tags=[]
    )
    a = await mem.experience("I work at Acme", user_id=USER)
    (old,) = await _facts(mem)
    await mem.storage.facts.update_fact_status(old.id, "promoted")
    extractor.extract.return_value = _extraction(
        "works_at", "Globex", category="identity", context="Works at Globex.", tags=[]
    )
    await mem.experience_result("I work at Globex", user_id=USER, supersedes=[a.id])

    (live,) = [f for f in await _facts(mem) if f.status != "archived"]
    assert (live.object, live.status, live.mention_count) == ("Globex", "promoted", 1)


async def test_a_retained_row_replaced_by_a_derived_fact_is_not_listed(
    mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    extractor.extract.return_value = _extraction(
        "lives_in", "Lisbon", category="identity", context="Lives in Lisbon.", tags=[]
    )
    a = await mem.experience("I live in Lisbon", user_id=USER)
    await mem.pin(a.id)
    extractor.extract.return_value = _extraction(
        "lives_in", "Porto", category="identity", context="Lives in Porto.", tags=[]
    )
    result = await mem.experience_result(
        "I live in Porto", user_id=USER, supersedes=[a.id]
    )

    # lives_in holds one value: the derived fact superseded the pinned row.
    assert result.superseded[0].retained_facts == []
    (live,) = await _facts(mem)
    assert (live.object, live.status) == ("Porto", "pinned")
