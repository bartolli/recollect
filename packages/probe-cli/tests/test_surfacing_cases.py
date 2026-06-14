"""Ground-truth-aware per-case classification (Axis 1, slice-1c)."""

from __future__ import annotations

from probe_cli.surfacing_cases import classify_cases
from probe_cli.surfacing_runner import QuerySurfacing, SurfacedFact


def _f(
    eval_id: str,
    score: float,
    *,
    relevant: bool,
    activated: bool = False,
    prop: float = 0.0,
    category: str = "",
) -> SurfacedFact:
    return SurfacedFact(
        fact_id="f",
        source_trace_id=eval_id,
        source_eval_id=eval_id,
        score=score,
        relevant=relevant,
        source_trace_activated=activated,
        source_trace_propagated_sim=prop,
        category=category,
    )


def _q(qid: str, facts: list[SurfacedFact]) -> QuerySurfacing:
    return QuerySurfacing(query_id=qid, surfaced=facts)


def _run(
    queries: list[QuerySurfacing],
    *,
    surface: dict[str, set[str]],
    forbid: dict[str, set[str]] | None = None,
    grouped: set[str] | None = None,
    floor: float = 0.65,
    top_k: int | None = None,
    activation_floor: float = 0.0,
):
    return classify_cases(
        queries,
        surface_by_query=surface,
        forbid_by_query=forbid or {},
        grouped=grouped or set(),
        floor=floor,
        top_k=top_k,
        activation_floor=activation_floor,
    )


def test_a_vector_above_floor_relevant() -> None:
    b = _run([_q("q", [_f("x", 0.80, relevant=True)])], surface={"q": {"x"}})
    assert b.a_vector == 1
    assert b.c_bridge == 0


def test_c_bridge_below_floor_nonbypass_activated() -> None:
    b = _run(
        [_q("q", [_f("x", 0.40, relevant=True, activated=True)])],
        surface={"q": {"x"}},
    )
    assert b.c_bridge == 1


def test_e_bypass_below_floor_relevant() -> None:
    b = _run(
        [_q("q", [_f("x", 0.40, relevant=True, activated=True, category="health")])],
        surface={"q": {"x"}},
    )
    assert b.e_bypass == 1
    assert b.c_bridge == 0


def test_f_readmit_nonrelevant_activated_below_floor() -> None:
    b = _run(
        [_q("d", [_f("y", 0.40, relevant=False, activated=True)])],
        surface={"d": set()},
    )
    assert b.f_readmit == 1
    assert b.clean is False


def test_g_violation_forbidden_surfaced() -> None:
    b = _run(
        [_q("q", [_f("wrong", 0.80, relevant=False)])],
        surface={"q": {"right"}},
        forbid={"q": {"wrong"}},
    )
    assert b.g_violation == 1


def test_misses_split_grouped_vs_ungrouped() -> None:
    b = _run([_q("q", [])], surface={"q": {"g1", "u1"}}, grouped={"g1"})
    assert b.c_miss == 1
    assert b.b_dropped == 1


def test_h_noise_nonrelevant_nonforbidden() -> None:
    b = _run(
        [_q("q", [_f("hay", 0.80, relevant=False)])],
        surface={"q": {"x"}},
    )
    assert b.h_noise == 1
    # h_noise is above-floor base-channel noise (surfaces today), so it does not
    # make the bridge verdict unclean.
    assert b.clean is True


def test_below_floor_nonactivated_nonrelevant_is_dropped_not_noise() -> None:
    # The floor gate: a below-floor, non-activated, non-bypass fact never
    # surfaces in production, so it is not noise -- it is silently dropped.
    b = _run([_q("q", [_f("x", 0.40, relevant=False)])], surface={"q": {"y"}})
    assert b.h_noise == 0
    assert b.f_readmit == 0
    assert b.clean is True


def test_below_floor_relevant_dropped_is_miss() -> None:
    # A relevant fact the floor drops (below, not activated, not bypass) is a
    # miss, classified grouped (c_miss) vs ungrouped (b_dropped).
    b = _run(
        [_q("q", [_f("g1", 0.40, relevant=True)])],
        surface={"q": {"g1"}},
        grouped={"g1"},
    )
    assert b.c_miss == 1
    assert b.c_bridge == 0


def test_bypass_offtarget_not_noise() -> None:
    # A bypass fact surfaces on an unrelated query by design (safety) -- it is
    # tracked as off-target, not a violation.
    b = _run(
        [_q("q", [_f("a", 0.40, relevant=False, category="health")])],
        surface={"q": {"x"}},
    )
    assert b.e_offtarget == 1
    assert b.h_noise == 0
    assert b.clean is True


def test_top_k_mirrors_production_rank_cut() -> None:
    # A noise fact ranked beyond top_k never surfaces in production.
    facts = [_f("x", 0.80, relevant=True)] + [
        _f(f"n{i}", 0.80, relevant=False) for i in range(5)
    ]
    b = _run([_q("q", facts)], surface={"q": {"x"}}, top_k=5)
    assert b.a_vector == 1
    assert b.h_noise == 4  # 5 noise facts, only 4 fit in top-5 after the needle


def test_g_split_bridge_vs_base_channel() -> None:
    # Above-floor forbidden = base channel (surfaces today); below-floor+activated
    # forbidden = bridge-caused cross-referent bleed.
    base = _q("q1", [_f("w", 0.80, relevant=False)])
    bridge = _q("q2", [_f("w", 0.40, relevant=False, activated=True)])
    b = _run(
        [base, bridge],
        surface={"q1": {"r"}, "q2": {"r"}},
        forbid={"q1": {"w"}, "q2": {"w"}},
    )
    assert b.g_violation == 2
    assert b.g_bridge == 1
    assert b.clean is False  # the bridge-caused one


def test_activation_floor_gates_weak_recovery() -> None:
    # Refined slice-2: a weak (prop < activation_floor) below-floor activation
    # does not recover the fact -- it becomes a miss; a strong one stays C.
    weak = _f("g1", 0.40, relevant=True, activated=True, prop=0.25)
    b = _run(
        [_q("q", [weak])],
        surface={"q": {"g1"}},
        grouped={"g1"},
        activation_floor=0.35,
    )
    assert b.c_bridge == 0
    assert b.c_miss == 1
    strong = _f("g1", 0.40, relevant=True, activated=True, prop=0.40)
    b2 = _run(
        [_q("q", [strong])],
        surface={"q": {"g1"}},
        grouped={"g1"},
        activation_floor=0.35,
    )
    assert b2.c_bridge == 1


def test_activation_floor_drops_weak_readmission() -> None:
    # A weak spurious activation on a distractor no longer re-admits.
    f = _f("y", 0.40, relevant=False, activated=True, prop=0.25)
    b = _run([_q("d", [f])], surface={"d": set()}, activation_floor=0.35)
    assert b.f_readmit == 0
    assert b.clean is True
