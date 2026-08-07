"""Revise-path reactivation of archived tokens.

Extend and revise are the two write-time edges that stamp traces into an
existing group; both must reactivate an archived target, else the stamp
feeds a token invisible to query-time activation (status='active' hop
filter). Extend twins live in test_v3_recall_tokens.py::TestTokenArchiving.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from recollect.core import CognitiveMemory
from recollect.llm.types import TokenAssessment
from recollect.models import MemoryTrace

_EMB_DIM = 768
_RELATED = MemoryTrace(
    id="related-1",
    content="Sarah is feeling better",
    embedding=[0.2 + i * 0.001 for i in range(_EMB_DIM)],
)
_ARCHIVED_GROUP = [
    {
        "token_id": "tok-archived",
        "label": "Sarah | old situation | old implication",
        "strength": 0.005,
        "significance": 0.7,
        "stamped_trace_ids": ["related-1"],
        "status": "archived",
    }
]
_ACTIVE_GROUP = [
    {
        "token_id": "tok-active",
        "label": "Sarah | current situation | current implication",
        "strength": 0.8,
        "significance": 0.6,
        "stamped_trace_ids": ["related-1"],
        "status": "active",
    }
]


def _setup(mock_storage, mock_extractor, assessment, *, groups=None):
    """Configure mocks for write-time token assessment tests."""
    mock_storage.vectors.search_semantic.return_value = [(_RELATED, 0.5)]
    mock_storage.recall_tokens.find_groups_for_traces = AsyncMock(
        return_value=groups or []
    )
    mock_extractor._provider = AsyncMock()
    mock_extractor._provider.complete_structured = AsyncMock(return_value=assessment)


@pytest.fixture()
def mem(mock_storage, mock_embeddings, mock_extractor):
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
        extractor=mock_extractor,
    )


class TestReviseReactivation:
    async def test_revise_reactivates_archived_token(
        self, mem, mock_storage, mock_extractor
    ):
        _setup(
            mock_storage,
            mock_extractor,
            TokenAssessment(
                action="revise",
                group_number=1,
                situation="risk resolved",
                implication="no action needed",
                significance=0.3,
            ),
            groups=_ARCHIVED_GROUP,
        )
        await mem.experience("Sarah's old risk turned out resolved", user_id="u1")
        mock_storage.recall_tokens.reinforce_tokens.assert_awaited()
        assert mock_storage.recall_tokens.reinforce_tokens.call_args[0][0] == [
            "tok-archived"
        ]
        # reinforce_tokens sets strength = significance on archived rows, so
        # the significance write must land first: strength = REVISED value.
        names = [name for name, _, _ in mock_storage.recall_tokens.mock_calls]
        assert names.index("update_token") < names.index("reinforce_tokens")

    async def test_revise_on_active_token_does_not_reinforce(
        self, mem, mock_storage, mock_extractor
    ):
        _setup(
            mock_storage,
            mock_extractor,
            TokenAssessment(
                action="revise",
                group_number=1,
                situation="updated situation",
                implication="new implication",
                significance=0.4,
            ),
            groups=_ACTIVE_GROUP,
        )
        await mem.experience("Sarah's situation changed again", user_id="u1")
        mock_storage.recall_tokens.reinforce_tokens.assert_not_awaited()
        mock_storage.recall_tokens.update_token.assert_awaited_once()
        args = mock_storage.recall_tokens.update_token.call_args[0]
        assert args[0] == "tok-active"
        assert args[2] == pytest.approx(0.4)
