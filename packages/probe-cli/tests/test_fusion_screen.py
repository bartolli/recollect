"""Fusion screen: concept rescales, entity-term preservation, cell diff, seed keys."""

from __future__ import annotations

from statistics import fmean, pstdev

import pytest
from probe_cli.corpus import TaskSeedTrace
from probe_cli.fusion_screen import (
    FusionForm,
    ScreenPreconditionError,
    cell_concepts,
    quantile_rescale,
    rescale_members,
    seed_keys,
    window_diff,
    zscore_rescale,
)

_CONCEPTS = {"a": 0.42, "b": 0.30, "c": 0.36, "z": 0.0}
_BASE = {"a": 0.61, "b": 0.74, "c": 0.55, "z": 0.70}
_MEMBERS = ("a", "b", "c")


def _order(values: dict[str, float]) -> list[str]:
    return sorted(_MEMBERS, key=values.__getitem__, reverse=True)


class TestZscoreRescale:
    def test_members_take_base_mean_and_std(self) -> None:
        out = zscore_rescale(_CONCEPTS, _BASE)
        rescaled = [out[t] for t in _MEMBERS]
        bases = [_BASE[t] for t in _MEMBERS]
        assert fmean(rescaled) == pytest.approx(fmean(bases))
        assert pstdev(rescaled) == pytest.approx(pstdev(bases))

    def test_concept_order_preserved(self) -> None:
        assert _order(zscore_rescale(_CONCEPTS, _BASE)) == _order(_CONCEPTS)

    def test_zero_concept_untouched(self) -> None:
        assert zscore_rescale(_CONCEPTS, _BASE)["z"] == 0.0

    def test_single_member_untouched(self) -> None:
        assert zscore_rescale({"a": 0.4}, {"a": 0.6}) == {"a": 0.4}


class TestQuantileRescale:
    def test_members_take_member_base_multiset(self) -> None:
        out = quantile_rescale(_CONCEPTS, _BASE)
        assert sorted(out[t] for t in _MEMBERS) == sorted(_BASE[t] for t in _MEMBERS)

    def test_concept_order_preserved(self) -> None:
        out = quantile_rescale(_CONCEPTS, _BASE)
        assert _order(out) == _order(_CONCEPTS)
        assert out["a"] == max(_BASE[t] for t in _MEMBERS)

    def test_ties_keep_input_order(self) -> None:
        out = quantile_rescale({"p": 0.3, "q": 0.3}, {"p": 0.2, "q": 0.9})
        assert out == {"p": 0.9, "q": 0.2}

    def test_zero_concept_untouched(self) -> None:
        assert quantile_rescale(_CONCEPTS, _BASE)["z"] == 0.0


_ACTIVATION_ONLY = frozenset({"x"})
_WITH_X = {**_CONCEPTS, "x": 0.50}
_BASE_X = {**_BASE, "x": 0.05}


class TestContentMembers:
    @pytest.mark.parametrize("form", ["zscore-content", "quantile-content"])
    def test_activation_only_keeps_raw_concept(self, form: FusionForm) -> None:
        out = rescale_members(
            _WITH_X, form=form, base=_BASE_X, activation_only=_ACTIVATION_ONLY
        )
        assert out["x"] == 0.50
        assert out["z"] == 0.0

    def test_zscore_statistics_over_content_members_only(self) -> None:
        out = rescale_members(
            _WITH_X,
            form="zscore-content",
            base=_BASE_X,
            activation_only=_ACTIVATION_ONLY,
        )
        assert [out[t] for t in _MEMBERS] == pytest.approx(
            [zscore_rescale(_CONCEPTS, _BASE)[t] for t in _MEMBERS]
        )

    def test_quantile_takes_content_member_bases(self) -> None:
        out = rescale_members(
            _WITH_X,
            form="quantile-content",
            base=_BASE_X,
            activation_only=_ACTIVATION_ONLY,
        )
        assert sorted(out[t] for t in _MEMBERS) == sorted(_BASE[t] for t in _MEMBERS)


class TestCellConcepts:
    def test_entity_term_preserved(self) -> None:
        rescaled = zscore_rescale(_CONCEPTS, _BASE)
        cell = cell_concepts(_CONCEPTS, rescaled=rescaled, entity_sims={"a": 0.8})
        assert cell.concept_sims["a"] == rescaled["a"]
        assert cell.entity_sims["a"] * cell.concept_sims["a"] == pytest.approx(
            0.8 * _CONCEPTS["a"]
        )

    def test_entity_without_concept_untouched(self) -> None:
        rescaled = zscore_rescale(_CONCEPTS, _BASE)
        cell = cell_concepts(_CONCEPTS, rescaled=rescaled, entity_sims={"z": 0.5})
        assert cell.entity_sims == {"z": 0.5}

    def test_nonpositive_rescale_drops_concept(self) -> None:
        cell = cell_concepts(
            {"a": 0.4, "b": 0.2},
            rescaled={"a": 0.5, "b": -0.1},
            entity_sims={"b": 0.9},
        )
        assert cell.dropped == ["b"]
        assert cell.entity_dropped == 1
        assert "b" not in cell.concept_sims
        assert cell.concept_sims["a"] == 0.5


class TestWindowDiff:
    def test_gained_and_lost_by_question(self) -> None:
        reference = {"q1": True, "q2": False, "q3": True, "q4": False}
        other = {"q1": True, "q2": True, "q3": False, "q4": False}
        diff = window_diff(reference, other)
        assert diff.gained == ["q2"]
        assert diff.lost == ["q3"]

    def test_identical_cells_differ_by_nothing(self) -> None:
        cells = {"q1": True, "q2": False}
        diff = window_diff(cells, dict(cells))
        assert diff.gained == []
        assert diff.lost == []


def _seed(seed_id: str, text: str) -> TaskSeedTrace:
    return TaskSeedTrace(id=seed_id, text=text, density_tier=0)


class TestSeedKeys:
    def test_maps_trace_to_seed_by_exact_content(self) -> None:
        seeds = [_seed("g1", "Maya moved to Lisbon."), _seed("g2", "Tom runs.")]
        keys = seed_keys({"t-1": "Tom runs.", "t-2": "Maya moved to Lisbon."}, seeds)
        assert keys == {"t-1": "g2", "t-2": "g1"}

    def test_unmapped_trace_fails(self) -> None:
        seeds = [_seed("g1", "Maya moved to Lisbon.")]
        with pytest.raises(ScreenPreconditionError, match="t-9"):
            seed_keys({"t-9": "Maya moved to Porto."}, seeds)
