"""Tests for Reciprocal Rank Fusion scoring."""

from __future__ import annotations

from recollect.core import _rrf_fuse


class TestRrfFuse:
    def test_single_source_ranking(self) -> None:
        ranked = {"semantic": ["a", "b", "c"]}
        scores = _rrf_fuse(ranked, k=60)
        # rank 1: 1/(60+1) = 0.01639..., rank 2: 1/(60+2), rank 3: 1/(60+3)
        assert scores["a"] > scores["b"] > scores["c"]
        assert abs(scores["a"] - 1.0 / 61) < 1e-9

    def test_multi_source_boosts_shared_items(self) -> None:
        ranked = {
            "semantic": ["a", "b", "c"],
            "entity": ["b", "a"],
        }
        scores = _rrf_fuse(ranked, k=60)
        # "a" rank 1 in semantic + rank 2 in entity = 1/61 + 1/62
        # "b" rank 2 in semantic + rank 1 in entity = 1/62 + 1/61
        # Both get the same total since ranks swap.
        assert abs(scores["a"] - scores["b"]) < 1e-9

    def test_item_in_more_sources_scores_higher(self) -> None:
        ranked = {
            "semantic": ["a", "b"],
            "entity": ["a", "c"],
            "activation": ["a", "d"],
        }
        scores = _rrf_fuse(ranked, k=60)
        # "a" appears in all 3 sources at rank 1: 3 * 1/(60+1)
        # "b", "c", "d" each appear in 1 source
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]
        assert scores["a"] > scores["d"]

    def test_empty_sources_returns_empty(self) -> None:
        scores = _rrf_fuse({}, k=60)
        assert scores == {}

    def test_custom_k_value(self) -> None:
        ranked = {"s1": ["a"]}
        scores_k1 = _rrf_fuse(ranked, k=1)
        scores_k100 = _rrf_fuse(ranked, k=100)
        # k=1: 1/(1+1) = 0.5; k=100: 1/(100+1) ~= 0.0099
        assert scores_k1["a"] > scores_k100["a"]
        assert abs(scores_k1["a"] - 0.5) < 1e-9

    def test_disjoint_sources(self) -> None:
        ranked = {
            "semantic": ["a", "b"],
            "entity": ["c", "d"],
        }
        scores = _rrf_fuse(ranked, k=60)
        assert set(scores.keys()) == {"a", "b", "c", "d"}
        # All items appear in exactly 1 source
        # Rank 1 items score higher than rank 2
        assert scores["a"] > scores["b"]
        assert scores["c"] > scores["d"]
        # Rank 1 from different sources are equal
        assert abs(scores["a"] - scores["c"]) < 1e-9
