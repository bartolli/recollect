"""Arm config loader."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from recollect.config import MemoryConfig


class ExtractionConfig(BaseModel):
    pydantic_ai_model: str = ""
    template_path: str = ""
    instructions: str = ""
    max_concepts: int = 5
    max_relations: int = 3
    max_tokens: int = 8192
    max_retries: int = 5
    model_settings: dict[str, Any] = Field(default_factory=dict)


class CorpusConfig(BaseModel):
    path: str


class OutputConfig(BaseModel):
    dir: str = "out"


class RetrievalConfig(BaseModel):
    enabled: bool = False
    db_url: str = ""
    traces_corpus_path: str = ""
    query_corpus_path: str = ""
    token_budget: int = Field(default=100_000, ge=1)
    top_k: int = Field(default=5, ge=1)
    max_retrievals_buffer: int = Field(default=50, ge=1)


class SituationalConfig(BaseModel):
    enabled: bool = False
    db_url: str = ""
    seed_traces_path: str = ""
    seed_groups_path: str = ""
    eval_corpus_path: str = ""


class SurfacingConfig(BaseModel):
    enabled: bool = False
    db_url: str = ""
    traces_corpus_path: str = ""
    query_corpus_path: str = ""


class Arm(BaseModel):
    name: str
    runs: int = Field(default=3, ge=1)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    corpus: CorpusConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    situational: SituationalConfig = Field(default_factory=SituationalConfig)
    surfacing: SurfacingConfig = Field(default_factory=SurfacingConfig)
    recollect_overrides: dict[str, Any] = Field(default_factory=dict)

    def to_memory_config(self) -> MemoryConfig:
        # Fresh instance per call — caller mutations must not bleed across arms.
        # recollect_overrides applied last so they win over arm extraction settings.
        cfg = MemoryConfig()
        cfg._set("extraction.pydantic_ai_model", self.extraction.pydantic_ai_model)
        cfg._set("extraction.template_path", self.extraction.template_path)
        cfg._set("extraction.instructions", self.extraction.instructions)
        cfg._set("extraction.max_concepts", self.extraction.max_concepts)
        cfg._set("extraction.max_relations", self.extraction.max_relations)
        cfg._set("extraction.max_tokens", self.extraction.max_tokens)
        cfg._set("extraction.max_retries", self.extraction.max_retries)
        cfg._set("extraction.model_settings", self.extraction.model_settings)
        if self.retrieval.enabled:
            cfg._set(
                "retrieval.max_retrievals", self.retrieval.max_retrievals_buffer
            )
            # database.url plumbed via connect(db_url) — PoolManager reads global cfg
        for path, value in self.recollect_overrides.items():
            cfg._set(path, value)
        return cfg


def load_arm(path: Path | str) -> Arm:
    p = Path(path)
    with p.open("rb") as f:
        data = tomllib.load(f)
    arm_block = data.get("arm", {})
    arm = Arm(
        name=arm_block.get("name", p.stem),
        runs=arm_block.get("runs", 3),
        extraction=ExtractionConfig(**data.get("extraction", {})),
        corpus=CorpusConfig(**data["corpus"]),
        output=OutputConfig(**data.get("output", {})),
        retrieval=RetrievalConfig(**data.get("retrieval", {})),
        situational=SituationalConfig(**data.get("situational", {})),
        surfacing=SurfacingConfig(**data.get("surfacing", {})),
        recollect_overrides=data.get("recollect_overrides", {}),
    )
    # Probe DB URLs reference env (.env) by name, never inlined in a committed
    # fixture; expandvars is a no-op on literal (no-$) values.
    arm.retrieval.db_url = os.path.expandvars(arm.retrieval.db_url)
    arm.situational.db_url = os.path.expandvars(arm.situational.db_url)
    arm.surfacing.db_url = os.path.expandvars(arm.surfacing.db_url)
    for db_url in (
        arm.retrieval.db_url,
        arm.situational.db_url,
        arm.surfacing.db_url,
    ):
        if "${" in db_url:
            raise ValueError(
                f"db_url has an unset env var: {db_url} -- set it in .env "
                "(see .env.example)"
            )
    return arm
