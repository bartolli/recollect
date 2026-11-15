"""Representations: every stored form of a statement and its fact lines after pin."""

from __future__ import annotations

import pytest
from probe_cli.corpus import CorpusEntry
from probe_cli.fusion_screen import ScreenPreconditionError
from probe_cli.representations import (
    FactLine,
    PinnedLine,
    RelationRow,
    StoredRelation,
    StoredState,
    StoredTrace,
    represent,
)
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import PersonaFact
from recollect_mcp.server import _safety_net_line


def _entry(cid: str) -> CorpusEntry:
    return CorpusEntry(id=cid, text=f"statement {cid}")


def _extraction(cid: str) -> ExtractionResult:
    return ExtractionResult(
        concepts=[f"concept {cid}"],
        relations=[Relation(source="user", relation="prefers", target=cid)],
    )


def _stored(tid: str, cid: str, *, status: str = "active") -> StoredTrace:
    dump = _extraction(cid).model_dump_json()
    return StoredTrace(id=tid, content=f"statement {cid}", status=status, pattern=dump)


def _fact(tid: str | None, target: str, **update: object) -> PersonaFact:
    fact = PersonaFact(
        subject="user",
        predicate="prefers",
        object=target,
        content=f"user prefers {target} -- hedged",
        source_trace_id=tid,
        category="preference",
    )
    return fact.model_copy(update=update)


class _Pins:
    def __init__(self, returns: dict[str, list[PersonaFact]]) -> None:
        self.returns = returns
        self.calls: list[str] = []

    async def __call__(self, trace_id: str) -> list[PersonaFact]:
        self.calls.append(trace_id)
        return self.returns.get(trace_id, [])


async def test_each_statement_carries_its_stored_representations() -> None:
    traces = [_stored("t-b", "b", status="archived"), _stored("t-a", "a")]
    relation = StoredRelation(
        source_trace_id="t-a",
        source_entity="user",
        relation="prefers",
        target_entity="a",
        context="said",
        weight=0.7,
    )
    cand = _fact("t-a", "a")
    prom = _fact("t-b", "b", status="promoted", category="health")
    facts = [cand, prom, _fact("t-x", "x")]
    state = StoredState(traces=traces, relations=[relation], facts=facts)
    a, b = await represent([_entry("a"), _entry("b")], state, pin=_Pins({}))
    assert (a.id, a.trace_content, a.trace_status) == ("a", "statement a", "active")
    assert (b.id, b.trace_status) == ("b", "archived")
    assert (a.extraction, b.extraction) == (_extraction("a"), _extraction("b"))
    triple = RelationRow(triple="user prefers a", context="said", weight=0.7)
    assert (a.entity_relations, b.entity_relations) == ([triple], [])
    assert a.persona_facts == [
        FactLine(line=_safety_net_line(cand), status="candidate", category="preference")
    ]
    assert b.persona_facts == [
        FactLine(line=_safety_net_line(prom), status="promoted", category="health")
    ]


async def test_pinned_lines_print_what_pin_returns_in_corpus_order() -> None:
    pa = _fact("t-a", "a", status="pinned")
    pb = [_fact("t-b", "b", status="pinned"), _fact("t-b", "b2", status="pinned")]
    pins = _Pins({"t-a": [pa], "t-b": pb})
    traces = [_stored("t-b", "b"), _stored("t-a", "a")]
    state = StoredState(traces=traces, relations=[], facts=[])
    a, b = await represent([_entry("a"), _entry("b")], state, pin=pins)
    assert pins.calls == ["t-a", "t-b"]
    assert a.pinned_fact_lines == [
        PinnedLine(line=_safety_net_line(pa), source="a", fallback=False)
    ]
    assert b.pinned_fact_lines == [
        PinnedLine(line=_safety_net_line(f), source="b", fallback=False) for f in pb
    ]


async def test_pinned_source_is_the_corpus_id_of_the_fact_trace() -> None:
    pins = _Pins({"t-a": [_fact("t-b", "a"), _fact("t-x", "x"), _fact(None, "y")]})
    traces = [_stored("t-a", "a"), _stored("t-b", "b"), _stored("t-x", "x")]
    state = StoredState(traces=traces, relations=[], facts=[])
    a, _ = await represent([_entry("a"), _entry("b")], state, pin=pins)
    assert [p.source for p in a.pinned_fact_lines] == ["b", "", ""]


_JUNK = _stored("t-b", "b").model_copy(update={"pattern": '{"significance": 7}'})


@pytest.mark.parametrize(
    ("b_traces", "refusal"),
    [
        ([], "^b: 0 traces"),
        ([_stored("t-b0", "b"), _stored("t-b1", "b")], "^b: 2 traces"),
        ([_JUNK], "^b: stored pattern is not an extraction"),
    ],
)
async def test_unreadable_statement_refuses_before_any_pin(
    b_traces: list[StoredTrace], refusal: str
) -> None:
    state = StoredState(traces=[_stored("t-a", "a"), *b_traces], relations=[], facts=[])
    pins = _Pins({})
    with pytest.raises(ScreenPreconditionError, match=refusal):
        await represent([_entry("a"), _entry("b")], state, pin=pins)
    assert pins.calls == []


async def test_a_statement_without_concepts_is_degraded() -> None:
    bare = ExtractionResult().model_dump_json()
    empty = StoredTrace(id="t-b", content="statement b", status="active", pattern=bare)
    state = StoredState(traces=[_stored("t-a", "a"), empty], relations=[], facts=[])
    a, b = await represent([_entry("a"), _entry("b")], state, pin=_Pins({}))
    assert (a.degraded, b.degraded) == (False, True)


async def test_the_pinned_statement_fallback_is_marked() -> None:
    noted = _fact("t-a", "statement a", predicate="noted", status="pinned")
    pins = _Pins({"t-a": [noted, _fact("t-a", "a", status="pinned")]})
    state = StoredState(traces=[_stored("t-a", "a")], relations=[], facts=[])
    (a,) = await represent([_entry("a")], state, pin=pins)
    assert [p.fallback for p in a.pinned_fact_lines] == [True, False]
