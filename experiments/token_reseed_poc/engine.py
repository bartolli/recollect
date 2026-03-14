import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import asyncpg
from pydantic import BaseModel, Field
from recollect.embeddings import FastEmbedProvider
from recollect.log_setup import logged

from experiments.token_reseed_poc.config import PocConfig

logger = logging.getLogger(__name__)


class TokenAssessment(BaseModel):
    action: str = "none"  # "create", "extend", "none"
    group_number: int = 0  # 1-based, which existing group to extend
    person_ref: str = ""  # "Alex", "Sarah (Alex's mother)"
    situation: str = ""  # "severe peanut allergy"
    implication: str = ""  # "EpiPen location and expiry"
    significance: float = 0.5  # 0.0-1.0, health/safety=high, trivia=low
    # 1-based indices, only for "create" action
    linked_indices: list[int] = Field(default_factory=list)


@dataclass
class StoredTrace:
    id: str
    content: str
    similarity: float = 0.0


@dataclass
class WriteTimeRecall:
    trace_content: str
    related_found: list[str]
    token_created: bool
    token_label: str = ""
    token_reused: bool = False
    token_revised: bool = False
    linked_indices: list[int] = field(default_factory=list)


@dataclass
class ExperienceResult:
    trace_id: str
    write_time_recall: WriteTimeRecall


class PocEngine:
    llm_errors: int = 0

    def __init__(self, config: PocConfig) -> None:
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._embedder = FastEmbedProvider(
            model_name=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        provider, _, model_name = config.llm.partition(":")
        if not model_name:
            msg = f"llm must be 'provider:model', got: {config.llm!r}"
            raise ValueError(msg)
        if provider == "ollama":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.ollama import OllamaProvider

            self._pai_model: Any = OpenAIChatModel(
                model_name, provider=OllamaProvider(base_url=config.llm_base_url)
            )
        else:
            self._pai_model = config.llm

        self._prompt_path: str | None = config.prompt_path

    async def setup(self) -> None:
        """Create pool, run schema, warm embeddings."""
        logger.info("Setting up PocEngine with database %s", self._config.database_url)
        self._pool = await asyncpg.create_pool(self._config.database_url)

        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text()
        async with self._pool.acquire() as conn:
            await conn.execute(schema_sql)
        logger.info("Schema applied")

        # Warm embeddings model
        await self._embedder.warm()
        logger.info("Embeddings model warmed")

    async def teardown(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PocEngine not set up; call setup() first")
        return self._pool

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return await self._embedder.generate_embedding(text)

    async def store(
        self,
        content: str,
        significance: float = 0.5,
        emotional_valence: float = 0.0,
    ) -> StoredTrace:
        """Embed and store content in poc_traces."""
        embedding = await self._embedder.generate_embedding(content)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO poc_traces"
                " (content, embedding, significance,"
                " emotional_valence)"
                " VALUES ($1, $2, $3, $4) RETURNING id",
                content,
                embedding_str,
                significance,
                emotional_valence,
            )
        trace_id = str(row["id"])
        logger.info("Stored trace %s: %s", trace_id[:8], content[:60])
        return StoredTrace(id=trace_id, content=content)

    async def find_related(
        self,
        embedding: list[float],
        exclude_id: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[StoredTrace]:
        """Vector search against poc_traces using cosine similarity."""
        if top_k is None:
            top_k = self._config.write_time_top_k
        if threshold is None:
            threshold = self._config.write_time_threshold

        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
                FROM poc_traces
                WHERE id != $2::uuid
                AND 1 - (embedding <=> $1::vector) > $3
                ORDER BY similarity DESC
                LIMIT $4
                """,
                embedding_str,
                exclude_id,
                threshold,
                top_k,
            )

        results = [
            StoredTrace(
                id=str(row["id"]),
                content=row["content"],
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]
        logger.info(
            "Found %d related traces for %s (threshold=%.2f)",
            len(results),
            exclude_id[:8],
            threshold,
        )
        return results

    def _load_prompt(
        self, new_content: str, numbered_list: str, existing_groups: str
    ) -> tuple[str, str]:
        """Load and format system/user prompts from file or use defaults."""
        if self._prompt_path is None:
            # Hardcoded default (situational group management)
            system_prompt = (
                "You manage situational groups for a memory system. "
                "Each group links memories about the same person/situation "
                "with specific implications. You can extend an existing group, "
                "create a new one, or do nothing. "
                "Respond with structured output only."
            )
            user_prompt = (
                f'New memory: "{new_content}"\n\n'
                f"Existing memories:\n{numbered_list}\n\n"
                f"Existing situational groups:\n{existing_groups}\n\n"
                f"Decide:\n"
                f"- extend: this memory adds a new implication to an existing group\n"
                f"- create: this memory starts a new group with some listed memories\n"
                f"- none: no situational connection\n"
            )
            return system_prompt, user_prompt

        raw = Path(self._prompt_path).read_text()
        # Strip comment lines (starting with #) and leading blank lines
        lines = raw.split("\n")
        content_lines = [line for line in lines if not line.startswith("#")]
        content = "\n".join(content_lines).strip()

        # Split on ---system--- and ---user--- markers
        parts = content.split("---system---")
        if len(parts) != 2:
            raise ValueError("Prompt file must contain exactly one ---system--- marker")
        after_system = parts[1]
        parts2 = after_system.split("---user---")
        if len(parts2) != 2:
            raise ValueError("Prompt file must contain exactly one ---user--- marker")

        system_prompt = parts2[0].strip()
        user_template = parts2[1].strip()
        user_prompt = user_template.format(
            new_content=new_content,
            numbered_list=numbered_list,
            existing_groups=existing_groups,
        )
        return system_prompt, user_prompt

    def _format_groups(self, groups: list[dict]) -> str:
        if not groups:
            return "None"
        lines = []
        for i, g in enumerate(groups, 1):
            indices_str = ", ".join(str(idx) for idx in g["memory_indices"])
            sig = g.get("significance", 0.5)
            lines.append(
                f"G{i}: {g['label']} (memories: {indices_str}, significance: {sig:.1f})"
            )
        return "\n".join(lines)

    @logged
    async def assess_situational_group(
        self,
        new_content: str,
        related: list[StoredTrace],
        existing_groups: list[dict],
    ) -> TokenAssessment:
        """Call LLM to assess situational group action for new memory."""
        numbered_list = "\n".join(
            f"{i + 1}. {trace.content}" for i, trace in enumerate(related)
        )
        groups_str = self._format_groups(existing_groups)

        system_prompt, user_prompt = self._load_prompt(
            new_content, numbered_list, groups_str
        )

        logger.info("Assessing situational group for: %s", new_content[:60])

        from pydantic_ai import Agent
        from pydantic_ai.exceptions import (
            ModelHTTPError,
            UnexpectedModelBehavior,
            UserError,
        )
        from pydantic_ai.settings import ModelSettings

        agent: Agent[None, TokenAssessment] = Agent(
            self._pai_model,
            system_prompt=system_prompt,
            output_type=TokenAssessment,
            retries=3,
        )
        try:
            result = await agent.run(
                user_prompt,
                model_settings=cast(
                    ModelSettings, {"max_tokens": self._config.llm_max_tokens}
                ),
            )
            assessment = result.output
        except (ModelHTTPError, UnexpectedModelBehavior, UserError):
            PocEngine.llm_errors += 1
            logger.warning("LLM assessment failed (logged: %d)", PocEngine.llm_errors)
            return TokenAssessment(action="none")

        logger.info(
            "Assessment: action=%s, group_number=%d, person_ref=%s, "
            "situation=%s, implication=%s, indices=%s",
            assessment.action,
            assessment.group_number,
            assessment.person_ref,
            assessment.situation,
            assessment.implication,
            assessment.linked_indices,
        )
        return assessment

    async def find_groups_for_traces(
        self,
        related: list[StoredTrace],
    ) -> list[dict]:
        """Find existing tokens stamped on related traces.

        Returns list of dicts:
            {
                "token_id": str,
                "label": str,
                "strength": float,
                "memory_indices": list[int],  # 1-based positions in related list
            }
        """
        if not related:
            return []
        related_ids = [r.id for r in related]
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT rt.id, rt.label, rt.strength, rt.significance,
                       array_agg(ts.trace_id::text) AS stamped_ids
                FROM poc_recall_tokens rt
                JOIN poc_token_stamps ts ON ts.token_id = rt.id
                WHERE ts.trace_id = ANY($1::uuid[])
                GROUP BY rt.id, rt.label, rt.strength, rt.significance
                ORDER BY rt.strength DESC
                """,
                related_ids,
            )

        # Map trace IDs back to 1-based positions in the related list
        id_to_index = {r.id: i + 1 for i, r in enumerate(related)}

        groups = []
        for row in rows:
            stamped = row["stamped_ids"]
            indices = sorted(id_to_index[tid] for tid in stamped if tid in id_to_index)
            if indices:
                groups.append(
                    {
                        "token_id": str(row["id"]),
                        "label": row["label"],
                        "strength": float(row["strength"]),
                        "significance": float(row["significance"]),
                        "memory_indices": indices,
                    }
                )

        logger.info(
            "Found %d existing groups for %d related traces",
            len(groups),
            len(related),
        )
        return groups

    async def update_token_label(self, token_id: str, new_label: str) -> None:
        """Update an existing token's label (e.g., to append new implications)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE poc_recall_tokens SET label = $1 WHERE id = $2::uuid",
                new_label,
                token_id,
            )
        logger.info("Updated token %s label to: %s", token_id[:8], new_label[:60])

    async def update_token(
        self, token_id: str, new_label: str, significance: float
    ) -> None:
        """Update an existing token's label and significance."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE poc_recall_tokens SET label = $1, significance = $2"
                " WHERE id = $3::uuid",
                new_label,
                significance,
                token_id,
            )
        logger.info(
            "Updated token %s: label=%s, significance=%.1f",
            token_id[:8],
            new_label[:60],
            significance,
        )

    async def stamp_trace(
        self,
        token_id: str,
        trace_id: str,
    ) -> None:
        """Stamp a single trace onto an existing token."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO poc_token_stamps (token_id, trace_id)
                VALUES ($1::uuid, $2::uuid)
                ON CONFLICT DO NOTHING
                """,
                token_id,
                trace_id,
            )
        logger.info(
            "Stamped trace %s onto existing token %s",
            trace_id[:8],
            token_id[:8],
        )

    async def create_token(
        self, label: str, trace_ids: list[str], significance: float = 0.5
    ) -> str:
        """Insert a recall token and stamp it onto the given traces."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO poc_recall_tokens (label, significance)"
                " VALUES ($1, $2) RETURNING id",
                label,
                significance,
            )
            token_id = str(row["id"])

            for trace_id in trace_ids:
                await conn.execute(
                    """
                    INSERT INTO poc_token_stamps (token_id, trace_id)
                    VALUES ($1::uuid, $2::uuid)
                    ON CONFLICT DO NOTHING
                    """,
                    token_id,
                    trace_id,
                )

        logger.info(
            "Created token %s (%s) stamped on %d traces",
            token_id[:8],
            label,
            len(trace_ids),
        )
        return token_id

    async def _extend_group(
        self,
        group: dict,
        trace_id: str,
        implication: str,
    ) -> tuple[str, list[int]]:
        """Extend an existing group: stamp trace and update label.

        Returns (final_label, memory_indices).
        """
        await self.stamp_trace(group["token_id"], trace_id)
        if implication:
            old_label = group["label"]
            if " | " in old_label:
                parts = old_label.split(" | ")
                if len(parts) >= 3:
                    parts[2] = parts[2] + ", " + implication
                else:
                    parts.append(implication)
                new_label = " | ".join(parts)
            else:
                new_label = old_label + " | " + implication
            await self.update_token_label(group["token_id"], new_label)
            final_label = new_label
        else:
            final_label = group["label"]
        logger.info(
            "Extended group %s with trace %s (implication: %s)",
            group["token_id"][:8],
            trace_id[:8],
            implication,
        )
        return final_label, group["memory_indices"]

    async def _revise_group(
        self,
        group: dict,
        trace_id: str,
        assessment: "TokenAssessment",
    ) -> tuple[str, list[int]]:
        """Revise an existing group: rebuild label from assessment, stamp trace.

        Returns (final_label, memory_indices).
        """
        old_label = group["label"]
        parts = old_label.split(" | ", 2)
        person_ref = parts[0] if len(parts) >= 1 else ""
        situation = (
            assessment.situation
            if assessment.situation
            else (parts[1] if len(parts) >= 2 else "")
        )
        implication = (
            assessment.implication
            if assessment.implication
            else (parts[2] if len(parts) >= 3 else "")
        )
        new_label = f"{person_ref} | {situation} | {implication}"
        await self.update_token(group["token_id"], new_label, assessment.significance)
        await self.stamp_trace(group["token_id"], trace_id)
        logger.info(
            "Revised group %s: %s -> %s",
            group["token_id"][:8],
            old_label[:40],
            new_label[:40],
        )
        return new_label, group["memory_indices"]

    async def experience(
        self,
        content: str,
        significance: float = 0.5,
        emotional_valence: float = 0.0,
    ) -> ExperienceResult:
        """Orchestrate: store, find related, find groups, assess,
        then create, extend, or revise token."""
        trace = await self.store(content, significance, emotional_valence)
        embedding = await self._embedder.generate_embedding(content)
        related = await self.find_related(embedding, exclude_id=trace.id)

        token_created = False
        token_reused = False
        token_revised = False
        token_label = ""
        linked_indices: list[int] = []

        if related:
            # Find existing situational groups on related memories
            existing_groups = await self.find_groups_for_traces(related)

            # Ask LLM
            assessment = await self.assess_situational_group(
                content, related, existing_groups
            )

            if assessment.action == "extend" and assessment.group_number > 0:
                group_idx = assessment.group_number - 1
                if 0 <= group_idx < len(existing_groups):
                    token_label, linked_indices = await self._extend_group(
                        existing_groups[group_idx],
                        trace.id,
                        assessment.implication,
                    )
                    token_created = True
                    token_reused = True

            elif assessment.action == "revise" and assessment.group_number > 0:
                group_idx = assessment.group_number - 1
                if 0 <= group_idx < len(existing_groups):
                    token_label, linked_indices = await self._revise_group(
                        existing_groups[group_idx],
                        trace.id,
                        assessment,
                    )
                    token_created = True
                    token_revised = True

            elif assessment.action == "create" and assessment.linked_indices:
                # Create new situational group
                existing_ids = []
                for idx in assessment.linked_indices:
                    if 1 <= idx <= len(related):
                        existing_ids.append(related[idx - 1].id)

                if existing_ids:
                    label = (
                        f"{assessment.person_ref} | "
                        f"{assessment.situation} | "
                        f"{assessment.implication}"
                    )
                    all_ids = [trace.id, *existing_ids]
                    await self.create_token(label, all_ids, assessment.significance)
                    token_created = True
                    token_label = label
                    linked_indices = assessment.linked_indices

        write_time_recall = WriteTimeRecall(
            trace_content=content,
            related_found=[t.content for t in related],
            token_created=token_created,
            token_label=token_label,
            token_reused=token_reused,
            token_revised=token_revised,
            linked_indices=linked_indices,
        )

        return ExperienceResult(
            trace_id=trace.id,
            write_time_recall=write_time_recall,
        )
