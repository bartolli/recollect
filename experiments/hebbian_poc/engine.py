import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import asyncpg
from pydantic import BaseModel, Field
from recollect.embeddings import FastEmbedProvider
from recollect.log_setup import logged

from experiments.hebbian_poc.config import PocConfig

logger = logging.getLogger(__name__)


class TokenAssessment(BaseModel):
    should_link: bool
    linked_indices: list[int] = []
    label: str = Field(default="", max_length=60)


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


@dataclass
class ExperienceResult:
    trace_id: str
    write_time_recall: WriteTimeRecall


class PocEngine:
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

    async def store(self, content: str) -> StoredTrace:
        """Embed and store content in poc_traces."""
        embedding = await self._embedder.generate_embedding(content)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO poc_traces (content, embedding)"
                " VALUES ($1, $2) RETURNING id",
                content,
                embedding_str,
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

    def _load_prompt(self, new_content: str, numbered_list: str) -> tuple[str, str]:
        """Load and format system/user prompts from file or use defaults."""
        if self._prompt_path is None:
            # Hardcoded default (relational.md equivalent)
            system_prompt = (
                "You identify whether memories refer to the same specific person, "
                "object, or ongoing situation. Your job is to GROUP memories that "
                "share a real-world referent -- not to find thematic similarity. "
                "Default to should_link=false. Respond with structured output only."
            )
            user_prompt = (
                f'New memory: "{new_content}"\n\n'
                f"Existing memories:\n{numbered_list}\n\n"
                f"Do any existing memories refer to the SAME specific person, object, "
                f"or ongoing situation as the new memory?\n\n"
                f"SAME referent (link these):\n"
                f'- "Sarah called about her cardiologist"'
                f' + "Sarah\'s blood pressure prescription"'
                f" (same Sarah, same health context)\n"
                f'- "Sofi\'s science fair project"'
                f' + "Sarah and Sofi building a volcano"'
                f" (same project)\n\n"
                f"DIFFERENT referents (do NOT link):\n"
                f'- "Alex\'s mother Sarah" + "Brian\'s daughter Sarah" '
                f"(different people named Sarah)\n"
                f'- "conference room Thursday" + "cardiologist Thursday" '
                f"(different events on same day)\n"
                f'- "Sofi\'s school" + "neighborhood block party" '
                f"(different contexts)\n\n"
                f'Key: two memories about "Sarah" only link if contextual clues '
                f"confirm they mean the SAME Sarah.\n\n"
                f"If same referent: should_link=true, list indices (1-based), "
                f"label under 5 words identifying the shared referent.\n"
                f"If different referents or unclear: should_link=false."
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
        )
        return system_prompt, user_prompt

    @logged
    async def assess_causal_link(
        self, new_content: str, related: list[StoredTrace]
    ) -> TokenAssessment:
        """Call LLM to assess whether memories are causally connected."""
        from pydantic_ai import Agent
        from pydantic_ai.exceptions import (
            ModelHTTPError,
            UnexpectedModelBehavior,
            UserError,
        )
        from pydantic_ai.settings import ModelSettings

        numbered_list = "\n".join(
            f"{i + 1}. {trace.content}" for i, trace in enumerate(related)
        )

        system_prompt, user_prompt = self._load_prompt(new_content, numbered_list)

        logger.info("Assessing causal link for: %s", new_content[:60])

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
            logger.warning("LLM assessment failed, defaulting to no link")
            return TokenAssessment(should_link=False)

        logger.info(
            "Assessment: should_link=%s, label=%s, indices=%s",
            assessment.should_link,
            assessment.label,
            assessment.linked_indices,
        )
        return assessment

    async def create_token(self, label: str, trace_ids: list[str]) -> str:
        """Insert a recall token and stamp it onto the given traces."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO poc_recall_tokens (label) VALUES ($1) RETURNING id",
                label,
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

    async def experience(self, content: str) -> ExperienceResult:
        """Orchestrate: store -> find_related -> assess -> optionally create token."""
        trace = await self.store(content)
        embedding = await self._embedder.generate_embedding(content)
        related = await self.find_related(embedding, exclude_id=trace.id)

        token_created = False
        token_label = ""

        if related:
            assessment = await self.assess_causal_link(content, related)

            if assessment.should_link and assessment.linked_indices:
                # Collect trace IDs: new trace + linked related
                # traces (1-based indices)
                linked_ids = [trace.id]
                for idx in assessment.linked_indices:
                    if 1 <= idx <= len(related):
                        linked_ids.append(related[idx - 1].id)

                await self.create_token(assessment.label, linked_ids)
                token_created = True
                token_label = assessment.label

        write_time_recall = WriteTimeRecall(
            trace_content=content,
            related_found=[t.content for t in related],
            token_created=token_created,
            token_label=token_label,
        )

        return ExperienceResult(
            trace_id=trace.id,
            write_time_recall=write_time_recall,
        )
