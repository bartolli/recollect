"""Weak-match screen: rule labels, cells over one capture, route split, cost."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from probe_cli import weak_match_screen
from probe_cli.arm import Arm, CorpusConfig, TaskConfig
from probe_cli.fusion_screen import SCREEN_DB, ScreenPreconditionError, _ScreenMemory
from probe_cli.weak_match_screen import (
    CELLS,
    ExpectedErrors,
    QueryCapture,
    RequiredLost,
    RuleCell,
    RuleName,
    ScreenQuery,
    check_template,
    cut_window,
    required_in_window,
    rule_cell,
    screen_cells,
    weak_traces,
    window_cost,
    window_lines,
)
from recollect.core import CognitiveMemory, _estimate_tokens
from recollect.models import MemoryTrace, Thought
from recollect_mcp.server import _format_thoughts

if TYPE_CHECKING:
    from pathlib import Path

    from probe_cli.corpus import TierLabel


def _trace(tid: str, relevance: float) -> Thought:
    text = f"memory {tid}"
    trace = MemoryTrace(id=tid, content=text)
    return Thought(trace=trace, relevance=relevance, reconstruction=text)


def _fact(tid: str, relevance: float) -> Thought:
    trace = MemoryTrace(id=tid, pattern={"persona_fact": True})
    return Thought(trace=trace, relevance=relevance, reconstruction=f"fact {tid}")


def _capture(
    window: list[Thought],
    *,
    qid: str = "q1",
    tier: TierLabel | None = None,
    required: str = "",
    tokens: dict[str, float] | None = None,
    sims: dict[str, float] | None = None,
    query: ScreenQuery | None = None,
) -> QueryCapture:
    return QueryCapture(
        query=query or ScreenQuery(id=qid, query="who"),
        window=window,
        token_bonuses=tokens or {},
        similarities=sims or {t.trace.id: 0.5 for t in window},
        tier_label=tier,
        required=frozenset({required} if required else ()),
    )


def _weak(capture: QueryCapture, rule: RuleName, parameter: float | None) -> set[str]:
    return set(weak_traces(capture, rule=rule, parameter=parameter))


def _cell(captures: list[QueryCapture], rule: RuleName, p: float | None) -> RuleCell:
    return rule_cell(captures, rule=rule, parameter=p, keys={})


def test_rule_labels() -> None:
    window = [_fact("f", 0.95), _trace("a", 0.80), _trace("b", 0.66)]
    window += [_trace("c", 0.61), _trace("d", 0.50)]
    capture = _capture(window)
    assert _weak(capture, "current", None) == set()
    assert _weak(_capture([_trace("e", 0.60)]), "fixed-floor", 0.60) == set()
    assert _weak(capture, "fixed-floor", 0.60) == {"d"}
    assert _weak(capture, "fixed-floor", 0.65) == {"c", "d"}
    assert _weak(capture, "fixed-floor", 0.70) == {"b", "c", "d"}
    # The top is the top trace: the fact at 0.95 is neither top nor labeled.
    assert _weak(capture, "top-relative", 0.15) == {"c", "d"}
    assert _weak(capture, "top-relative", 0.20) == {"d"}
    assert _weak(_capture([_trace("a", 0.40)]), "top-relative", 0.10) == set()
    # Store 0.40..0.80: median 0.60, pstdev 0.1414.
    sims = {"x": 0.40, "y": 0.50, "z": 0.60, "a": 0.80, "b": 0.70}
    store = _capture([_trace("a", 0.9), _trace("b", 0.9)], sims=sims)
    assert _weak(store, "store-relative", 1.0) == {"b"}
    assert _weak(store, "store-relative", 1.5) == {"a", "b"}
    flat = _capture([_trace("a", 0.9)], sims={"a": 0.5, "x": 0.5})
    assert _weak(flat, "store-relative", 2.0) == set()


def _fail(*_: object, **__: object) -> None:
    raise AssertionError("retrieval ran while computing cells")


def test_rules_read_one_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CognitiveMemory, "think_about", _fail)
    monkeypatch.setattr(_ScreenMemory, "replay", _fail)
    window = [_trace("a", 0.82), _fact("f", 0.70), _trace("b", 0.58)]
    captures = [_capture(window, sims={"a": 0.9, "b": 0.2, "x": 0.3})]
    cells = screen_cells(captures, keys={"a": "g1"})
    assert [(c.rule, c.parameter) for c in cells] == list(CELLS)
    unlabeled = window_lines(window, keys={"a": "g1"})
    assert unlabeled == ["1. g1 0.820", "2. ~f 0.700", "3. b 0.580"]
    for cell in cells:
        (query,) = cell.queries
        assert all(map(str.startswith, query.window, unlabeled))
        assert query.cost_current == window_cost(window)


def test_required_lost_splits_token_and_direct() -> None:
    rows = [("r1", 0.55, "t3", 0.3), ("r2", 0.58, "t2", 0.0), ("r3", 0.90, "t1", 0.0)]
    captures = [
        _capture([_trace(r, rel)], qid=f"q{i}", tier=tier, required=r, tokens={r: b})
        for i, (r, rel, tier, b) in enumerate(rows, start=1)
    ]
    captures.append(_capture([_trace("o", 0.40)], qid="q4", tier="t3", required="r4"))
    assert required_in_window(captures) == ["q1", "q2", "q3"]
    lost = _cell(captures, "fixed-floor", 0.60).required_lost
    assert (lost.token, lost.direct) == (["q1"], ["q2"])
    assert lost.by_tier == {"t3": 1, "t2": 1}
    assert _cell(captures, "current", None).required_lost == RequiredLost()


def test_none_tier_counts_strong_traces() -> None:
    window = [_fact("f", 0.9), _trace("a", 0.70), _trace("b", 0.62), _trace("c", 0.5)]
    captures = [
        _capture(window, qid="abq01", tier="none"),
        _capture([_trace("d", 0.3)], qid="q2", tier="t1"),
    ]
    floor = _cell(captures, "fixed-floor", 0.65)
    assert floor.none_tier_strong == {"abq01": 1}
    assert floor.none_tier_strong_total == 1
    assert _cell(captures, "current", None).none_tier_strong_total == 3


def test_cost_uses_the_mcp_formatter() -> None:
    strong, weak, fact = _trace("s", 0.80), _trace("w", 0.50), _fact("f", 0.90)
    window = [fact, strong, weak]
    (query,) = _cell([_capture(window)], "fixed-floor", 0.60).queries
    text = _format_thoughts(window, safety_facts=())
    cut_text = _format_thoughts(cut_window(window, frozenset({"w"})), safety_facts=())
    assert query.cost_current == _estimate_tokens(text)
    assert query.cost_cut == _estimate_tokens(cut_text)
    assert "memory w" in text and "memory w" not in cut_text
    assert "memory s" in cut_text and "fact f" in cut_text
    assert query.cost_cut < query.cost_current


def test_expected_ids_report_both_errors() -> None:
    query = ScreenQuery(
        id="kids",
        query="children",
        expected_strong=["aaaa1111", "cccc3333"],
        expected_weak=["bbbb2222", "dddd4444"],
    )
    window = [_trace("aaaa1111-x", 0.55), _trace("bbbb2222-y", 0.90)]
    window.append(_trace("eeee5555-z", 0.70))
    (cell,) = _cell([_capture(window, query=query)], "fixed-floor", 0.60).queries
    assert cell.window[0] == "1. aaaa1111 0.550 weak"
    assert cell.expected == ExpectedErrors(
        strong_labeled_weak=["aaaa1111"],
        weak_left_strong=["bbbb2222"],
        not_delivered=["cccc3333", "dddd4444"],
    )


async def test_refuses_missing_or_self_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    check_template("probe_gate0_d2_keep", {"probe_gate0_d2_keep"})
    copies: list[str] = []

    async def _names(_: str) -> set[str]:
        return {"probe_gate0_d2_keep", SCREEN_DB}

    async def _copy(_: str, template: str) -> None:
        copies.append(template)

    monkeypatch.setattr(weak_match_screen, "_database_names", _names)
    monkeypatch.setattr(weak_match_screen, "_copy_store", _copy)
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"id": "q1", "query": "who"}\n', encoding="utf-8")
    task = TaskConfig(enabled=True, db_url="postgresql://probe@localhost/probe_task")
    arm = Arm(name="screen", corpus=CorpusConfig(path=""), task=task)
    for template in ("probe_absent", SCREEN_DB):
        with pytest.raises(ScreenPreconditionError, match=template):
            await weak_match_screen.screen(
                arm, template=template, user_id="u", queries_path=queries
            )
    assert copies == []
