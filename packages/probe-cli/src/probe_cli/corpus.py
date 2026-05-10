"""JSONL corpus readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from recollect.llm.types import Predicate
from recollect.models import FactCategory

ExpectedAction = Literal["create", "extend", "revise", "none"]


class CorpusEntry(BaseModel):
    id: str
    text: str
    expected_category: FactCategory | None = None
    expected_predicate: Predicate | None = None
    notes: str = ""


class Corpus(BaseModel):
    entries: list[CorpusEntry] = Field(default_factory=list)
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.entries)


class QueryEntry(BaseModel):
    # relevant_trace_ids references CorpusEntry.id; empty list = distractor query.
    id: str
    text: str
    relevant_trace_ids: list[str] = Field(default_factory=list)
    expected_category: FactCategory | None = None
    phrasing_style: str = ""
    notes: str = ""


class QueryCorpus(BaseModel):
    entries: list[QueryEntry] = Field(default_factory=list)
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.entries)


def _load_jsonl[T: BaseModel](
    path: Path | str, model: type[T], label: str
) -> tuple[list[T], str]:
    p = Path(path)
    items: list[T] = []
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data = json.loads(stripped)
            try:
                items.append(model.model_validate(data))
            except Exception as exc:
                raise ValueError(
                    f"{p}:{line_no}: invalid {label}: {exc}"
                ) from exc
    return items, str(p)


def load_corpus(path: Path | str) -> Corpus:
    entries, src = _load_jsonl(path, CorpusEntry, "corpus entry")
    return Corpus(entries=entries, source_path=src)


def load_query_corpus(path: Path | str) -> QueryCorpus:
    entries, src = _load_jsonl(path, QueryEntry, "query entry")
    return QueryCorpus(entries=entries, source_path=src)


class SeedTrace(BaseModel):
    id: str
    text: str
    expected_category: FactCategory | None = None
    seed_group_id: str | None = None


class SeedGroup(BaseModel):
    group_id: str
    person_ref: str
    situation: str
    implications: list[str] = Field(default_factory=list, min_length=1)
    significance: float = Field(ge=0.0, le=1.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    status: Literal["active", "archived"] = "active"
    member_trace_ids: list[str] = Field(default_factory=list, min_length=1)

    @property
    def label(self) -> str:
        # Persisted shape: "person_ref | situation | impl1, impl2, ..."
        return f"{self.person_ref} | {self.situation} | {', '.join(self.implications)}"


class EvalEntry(BaseModel):
    id: str
    text: str
    expected_category: FactCategory | None = None
    expected_action: ExpectedAction
    expected_group_id: str | None = None
    expected_linked_trace_ids: list[str] = Field(default_factory=list)
    expected_person_ref: str = ""
    expected_situation: str = ""
    expected_implication: str = ""
    expected_significance: float | None = Field(default=None, ge=0.0, le=1.0)
    chain: str = ""
    notes: str = ""


def load_seed_traces(path: Path | str) -> list[SeedTrace]:
    entries, _ = _load_jsonl(path, SeedTrace, "seed trace")
    return entries


def load_seed_groups(path: Path | str) -> list[SeedGroup]:
    entries, _ = _load_jsonl(path, SeedGroup, "seed group")
    return entries


def load_eval_corpus(path: Path | str) -> list[EvalEntry]:
    entries, _ = _load_jsonl(path, EvalEntry, "eval entry")
    return entries
