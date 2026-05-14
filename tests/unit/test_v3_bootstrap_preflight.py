"""Unit tests for bootstrap.preflight.

preflight() returns a list of InvariantCheck results. Strict mode raises
on any failure listing all of them. Currently a single check:
embedding_contract_matches (delegates to verify_embedding_contract).
"""

from __future__ import annotations

import pytest
from recollect.bootstrap.preflight import (
    InvariantCheck,
    preflight_embedding_contract,
    raise_on_failures,
)
from recollect.exceptions import BootstrapError


class TestEmbeddingContractCheck:
    def test_match_returns_ok(self) -> None:
        result = preflight_embedding_contract(
            stored=("nomic-ai/nomic-embed-text-v1.5", "v0.7"),
            current=("nomic-ai/nomic-embed-text-v1.5", "v0.7"),
        )
        assert result.ok is True
        assert result.name == "embedding_contract_matches"

    def test_missing_stored_returns_ok(self) -> None:
        # Pre-m003 DB: no row yet -> treated as compatible.
        result = preflight_embedding_contract(
            stored=None,
            current=("nomic-ai/nomic-embed-text-v1.5", "v0.7"),
        )
        assert result.ok is True

    def test_model_mismatch_returns_fail(self) -> None:
        result = preflight_embedding_contract(
            stored=("old-model", "v0.7"),
            current=("nomic-ai/nomic-embed-text-v1.5", "v0.7"),
        )
        assert result.ok is False
        assert "old-model" in result.detail

    def test_prefix_mismatch_returns_fail(self) -> None:
        result = preflight_embedding_contract(
            stored=("nomic-ai/nomic-embed-text-v1.5", "v0.6"),
            current=("nomic-ai/nomic-embed-text-v1.5", "v0.7"),
        )
        assert result.ok is False
        assert "v0.6" in result.detail


class TestRaiseOnFailures:
    def test_all_ok_returns_none(self) -> None:
        results = [InvariantCheck("a", ok=True, detail="ok")]
        raise_on_failures(results)  # no raise

    def test_any_failure_raises_with_all_failures_listed(self) -> None:
        results = [
            InvariantCheck("a", ok=True, detail="ok"),
            InvariantCheck("b", ok=False, detail="bad-b"),
            InvariantCheck("c", ok=False, detail="bad-c"),
        ]
        with pytest.raises(BootstrapError, match=r"bad-b.*bad-c"):
            raise_on_failures(results)
