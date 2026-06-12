"""Unit tests for FastEmbedProvider.contract() + verify_embedding_contract.

Pure unit: no DB, no real model load. Asserts contract shape (model,
task_prefix_version) and the verify-against-stored guard semantics.
"""

from __future__ import annotations

import pytest
from recollect.config import MemoryConfig
from recollect.embeddings import FastEmbedProvider
from recollect.exceptions import EmbeddingContractError
from recollect.storage_ops import verify_embedding_contract


class TestProviderContract:
    def test_contract_returns_model_and_prefix_version(self) -> None:
        provider = FastEmbedProvider()
        contract = provider.contract()
        assert isinstance(contract, tuple)
        assert len(contract) == 2
        model, version = contract
        assert model == "nomic-ai/nomic-embed-text-v1.5"
        assert version == FastEmbedProvider.TASK_PREFIX_VERSION

    def test_custom_model_reflects_in_contract(self) -> None:
        provider = FastEmbedProvider(model_name="custom-model")
        model, _ = provider.contract()
        assert model == "custom-model"

    def test_from_config_contract_reflects_configured_model(self) -> None:
        cfg = MemoryConfig()
        cfg._set("embedding.model", "custom/from-config-model")
        provider = FastEmbedProvider.from_config(cfg)
        assert provider.contract() == (
            "custom/from-config-model",
            FastEmbedProvider.TASK_PREFIX_VERSION,
        )


class TestVerifyEmbeddingContract:
    def test_matching_contract_passes(self) -> None:
        provider = FastEmbedProvider()
        c = provider.contract()
        verify_embedding_contract(stored=c, current=c)

    def test_missing_stored_passes_as_uninitialized(self) -> None:
        # Pre-m003 DB: no row yet -> stored is None, treated as compatible.
        # m003 will insert on first bootstrap; subsequent connects then match.
        provider = FastEmbedProvider()
        verify_embedding_contract(stored=None, current=provider.contract())

    def test_model_mismatch_raises(self) -> None:
        version = FastEmbedProvider.TASK_PREFIX_VERSION
        with pytest.raises(EmbeddingContractError, match="model"):
            verify_embedding_contract(
                stored=("old-model", version),
                current=("nomic-ai/nomic-embed-text-v1.5", version),
            )

    def test_prefix_version_mismatch_raises(self) -> None:
        with pytest.raises(EmbeddingContractError, match="task_prefix_version"):
            verify_embedding_contract(
                stored=("nomic-ai/nomic-embed-text-v1.5", "v0.6"),
                current=("nomic-ai/nomic-embed-text-v1.5", "v0.7"),
            )
