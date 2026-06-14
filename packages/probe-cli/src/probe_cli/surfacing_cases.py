"""Ground-truth-aware per-case classification for the associative corpus.

Maps each surfaced persona fact to its outcome case (DESIGN.md taxonomy, Axis 1)
against the ground-truth surface/forbid sets.

The probe scores the pre-floor pool (_find_relevant_persona_facts returns the
top max_facts_per_query regardless of floor), so the metric applies the
production floor gate itself -- a fact "surfaces" iff above-floor OR bypass OR
token-activated. A below-floor non-bypass non-activated fact is dropped, not
noise. A case is the outcome the run assigns, not an authored label.

c_miss/b_dropped count expected surfaces that did not surface post-gate; that
conflates "floor dropped it" with "no fact extracted", so read them with the
substrate (extraction) caveat, not as a clean recall number.
"""

from __future__ import annotations

from pydantic import BaseModel

from probe_cli.surfacing_runner import QuerySurfacing, SurfacedFact

_BYPASS: frozenset[str] = frozenset({"health", "dietary"})


class CaseBreakdown(BaseModel):
    a_vector: int = 0       # relevant, above floor (vector surfaced)
    c_bridge: int = 0       # relevant, below floor, non-bypass, activated
    e_bypass: int = 0       # relevant, below floor, bypass (surfaces regardless)
    anomaly: int = 0        # relevant, surfaced, none of the above (unexpected)
    c_miss: int = 0         # expected grouped surface that did not surface
    b_dropped: int = 0      # expected ungrouped surface that did not surface
    f_readmit: int = 0      # non-relevant, activated, below floor, non-bypass
    g_violation: int = 0    # forbidden (wrong-referent) fact surfaced (total)
    g_bridge: int = 0       # forbidden subset caused by the bridge (below+activated)
    h_noise: int = 0        # non-relevant, above floor (vector precision miss)
    e_offtarget: int = 0    # non-relevant bypass fact surfaced (expected safety)

    @property
    def clean(self) -> bool:
        # No bridge-caused harm: no re-admission, no bridge-caused cross-referent
        # bleed. h_noise and the above-floor part of g_violation are base-channel
        # (surface today without slice-2), so they do not gate the bridge verdict.
        return self.f_readmit == 0 and self.g_bridge == 0


def _surfaces(f: SurfacedFact, floor: float, activation_floor: float) -> bool:
    # Refined slice-2: a below-floor fact recovers only via STRONG activation
    # (propagated_sim >= activation_floor) -- weak incidental activation does not
    # surface it. activation_floor=0 reproduces the bare "any activation" bridge.
    activated = (
        f.source_trace_activated
        and f.source_trace_propagated_sim >= activation_floor
    )
    return f.score >= floor or f.category in _BYPASS or activated


def _relevant_case(f: SurfacedFact, floor: float) -> str:
    if f.score >= floor:
        return "a_vector"
    if f.category in _BYPASS:
        return "e_bypass"
    return "c_bridge" if f.source_trace_activated else "anomaly"


def _noise_case(f: SurfacedFact, *, in_forbid: bool, floor: float) -> str:
    if in_forbid:
        return "g_violation"
    if f.source_trace_activated and f.score < floor and f.category not in _BYPASS:
        return "f_readmit"
    if f.category in _BYPASS:
        return "e_offtarget"
    return "h_noise"


def classify_cases(
    query_results: list[QuerySurfacing],
    *,
    surface_by_query: dict[str, set[str]],
    forbid_by_query: dict[str, set[str]],
    grouped: set[str],
    floor: float,
    top_k: int | None = None,
    activation_floor: float = 0.0,
) -> CaseBreakdown:
    # top_k mirrors production's _rank_and_limit_facts cut (q.surfaced is already
    # in _compute_fact_relevance order); None = the full statistical cohort.
    # activation_floor gates the bridge on propagated_sim (refined slice-2).
    counts: dict[str, int] = dict.fromkeys(CaseBreakdown.model_fields, 0)
    for q in query_results:
        surface = surface_by_query.get(q.query_id, set())
        forbid = forbid_by_query.get(q.query_id, set())
        ranked = q.surfaced[:top_k] if top_k is not None else q.surfaced
        surfaced_relevant: set[str] = set()
        for f in ranked:
            if not _surfaces(f, floor, activation_floor):
                continue  # floor-dropped: not noise; a relevant drop is a miss
            forbidden = f.source_eval_id in forbid
            if f.source_eval_id in surface:
                surfaced_relevant.add(f.source_eval_id)
                counts[_relevant_case(f, floor)] += 1
            else:
                counts[_noise_case(f, in_forbid=forbidden, floor=floor)] += 1
            if forbidden and f.score < floor and f.category not in _BYPASS:
                counts["g_bridge"] += 1
        for ev in surface - surfaced_relevant:
            counts["c_miss" if ev in grouped else "b_dropped"] += 1
    return CaseBreakdown(**counts)
