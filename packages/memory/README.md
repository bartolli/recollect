# recollect

Persistent memory SDK for LLM agents. See the [project README](https://github.com/bartolli/recollect#readme) for architecture and scoring details.

## Install

```bash
pip install recollect    # or: uv add recollect
```

## Quick Start

```python
import asyncio
from recollect import CognitiveMemory

async def main():
    memory = CognitiveMemory()
    await memory.connect()

    await memory.experience(
        "The team decided to migrate from Redis to PostgreSQL for persistence."
    )

    thoughts = await memory.think_about("database decisions", token_budget=500)
    for thought in thoughts:
        print(f"[{thought.activation:.2f}] {thought.content}")

    await memory.close()

asyncio.run(main())
```

## API

| Method | Description |
|--------|-------------|
| `connect(db_url=None)` | Connect to PostgreSQL. Uses `DATABASE_URL` env var if no argument. |
| `experience(content)` | Store a memory trace. LLM extracts entities, concepts, significance. |
| `think_about(query, token_budget)` | Retrieve memories that fit within a token limit. Returns `list[Thought]`. |
| `consolidate(threshold=None)` | Merge and prune weak traces. |
| `forget(trace_id)` | Remove a trace. |
| `reinforce(trace_id, factor=1.1)` | Strengthen a trace. |
| `facts(subject=None)` | List persona facts. |
| `start_session(user_id)` | Begin a scoped session. |
| `close()` | Disconnect and release resources. |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | `postgresql://localhost:5432/memory_sdk` | PostgreSQL connection string. |
| `PYDANTIC_AI_MODEL` | No | -- | pydantic-ai model string in `provider:model` format (e.g., `ollama:ministral-3`, `anthropic:claude-haiku-4-5-20251001`). |
| `ANTHROPIC_API_KEY` | For Anthropic models | -- | Anthropic API key. Read by pydantic-ai's Anthropic backend. |
| `OPENAI_API_KEY` | For OpenAI models | -- | OpenAI API key. Read by pydantic-ai's OpenAI backend. |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama API endpoint. |
| `MEMORY_EXTRACTION_MAX_TOKENS` | No | `8192` | Max tokens for LLM extraction. Reasoning models consume thinking tokens before output; 8192 covers most cases. |
| `MEMORY_CONFIG` | No | -- | Path to custom TOML config file. |
| `MEMORY_EXTRACTION_INSTRUCTIONS` | No | -- | Override extraction prompt instructions. |
| `MEMORY_RECALL_TOKENS_ENABLED` | No | `true` | Enable write-time token stamping and query-time activation. |
| `MEMORY_RECALL_TOKENS_TOP_K` | No | `5` | Max related traces to consider for token assessment. |
| `MEMORY_RECALL_TOKENS_THRESHOLD` | No | `0.42` | Min cosine similarity to consider a trace as related at write time. |
| `MEMORY_RECALL_TOKENS_STRENGTH_THRESHOLD` | No | `0.1` | Min token strength to activate at query time. |
| `MEMORY_RECALL_TOKENS_SCORE_BONUS` | No | `0.1` | Gated additive bonus: `token_strength * bonus * effective_sim`. |
| `MEMORY_RECALL_TOKENS_REINFORCE_BOOST` | No | `0.1` | Strength increment on token activation (capped at 1.0). |
| `MEMORY_RECALL_TOKENS_DECAY_FACTOR` | No | `0.9` | Multiply inactive token strength by this during consolidation. |

## Configuration

```toml
[memory]
decay_rate = 0.05

[retrieval]
max_retrievals = 10

[extraction]
pydantic_ai_model = "ollama:ministral-3"   # pydantic-ai provider:model format
```

### Config sections

| Section | Controls | Key parameters |
|---------|----------|----------------|
| `[database]` | PostgreSQL connection | `url` |
| `[memory]` | Core memory model | `initial_strength`, `consolidation_threshold`, `decay_rate` |
| `[working_memory]` | Working memory capacity | `capacity` (default 7, range 5-9) |
| `[retrieval]` | Retrieval pipeline tuning | `max_retrievals`, `search_limit`, `selection_threshold` |
| `[extraction]` | LLM extraction | `max_tokens`, `max_concepts`, `max_relations`, `pydantic_ai_model` |
| `[embedding]` | Local embedding model | `model`, `dimensions` |
| `[persona]` | Persona fact management | `auto_extract`, `confidence_threshold` |
| `[session]` | Session summaries | `summary_strength`, `summary_max_tokens` |

Full defaults: [`config.toml`](https://github.com/bartolli/recollect/blob/main/packages/memory/src/recollect/config.toml)

```python
from recollect.config import MemoryConfig

config = MemoryConfig(config_path=Path("./my-config.toml"))
memory = CognitiveMemory(config=config)
```

## LLM Provider

```python
from recollect.llm.pydantic_ai import PydanticAIProvider

# Model configured via PYDANTIC_AI_MODEL env var, or pass explicitly:
provider = PydanticAIProvider()  # uses PYDANTIC_AI_MODEL
provider = PydanticAIProvider(model="anthropic:claude-sonnet-4-6")
provider = PydanticAIProvider(model="ollama:llama3")
```

### Reasoning models

Models that use internal chain-of-thought (OpenAI o1/o3, Qwen3, DeepSeek-R1) consume thinking tokens from the `max_tokens` budget. If extraction returns empty responses, increase the token budget:

```toml
# memory.toml
[extraction]
max_tokens = 8192
```

The default is 8192 to accommodate thinking tokens. Non-reasoning models work fine at this budget; no need to reduce it.

## Requirements

- Python 3.12+
- PostgreSQL 17 with [pgvector](https://github.com/pgvector/pgvector)
- `DATABASE_URL` environment variable

## License

MIT
