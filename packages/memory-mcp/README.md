# recollect-mcp

MCP server for persistent memory. 6 tools, 3 resources, server-managed sessions. See the [project README](https://github.com/bartolli/recollect#readme) for architecture details.

## Install

```bash
pip install recollect-mcp    # or: uv add recollect-mcp
```

## Usage

```bash
# stdio (default)
recollect-mcp

# streamable-http
recollect-mcp --transport streamable-http

# with logging
recollect-mcp --log-file logs/mcp.jsonl --verbose
```

## Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `remember` | `content: str` | Store an experience. LLM extracts entities, concepts, significance, and persona facts. |
| `recall` | `query: str`, `token_budget: int = 2000` | Retrieve relevant memories. Returns persona facts as context followed by matching traces. |
| `reflect` | -- | Load persona context for the current session. Call before responding to any user message. |
| `pin` | `trace_id: str` | Promote a memory's extracted relations to permanent persona facts. |
| `unpin` | `fact_id: str` | Archive a persona fact. It stops surfacing in recall and reflect; the row is retained. |
| `forget` | `trace_id: str`, `force: bool = false` | Forget a memory trace: it stops surfacing in recall and never auto-revives. Derived facts archive; safety-critical (health/dietary/constraint) and pinned facts are retained unless `force=true`. Nothing is hard-deleted. |

## Resources

| URI | Description |
|-----|-------------|
| `memory://primer` | Relational graph of persona facts. Read at conversation start for user context. |
| `memory://facts` | All active persona facts with confidence scores and timestamps. |
| `memory://health` | Server and database health status. |

Clients that support MCP resources get session priming automatically via `primer`. For clients that don't, `reflect` loads the same context as a tool call. If neither is invoked, the server injects the primer on the first tool call of the session.

## Client configuration

Add to `.mcp.json` (Claude Code) or `claude_desktop_config.json` (Claude Desktop). Keep API keys out of the JSON: put them in an env file and pass it with `--env-file`. Use an absolute path -- the client spawns the server from its own working directory.

```json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": [
        "--env-file",
        "/Users/you/.config/recollect/recollect.env",
        "recollect-mcp"
      ],
      "env": {
        "MEMORY_USER_ID": "your-user-id",
        "DATABASE_URL": "postgresql://user@localhost:5432/dbname",
        "PYDANTIC_AI_MODEL": "anthropic:claude-haiku-4-5-20251001"
      }
    }
  }
}
```

```bash
# /Users/you/.config/recollect/recollect.env -- secrets only, chmod 600
ANTHROPIC_API_KEY=sk-ant-...
```

Variables in the `env` block take precedence over the env file, so define each in one place only.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MEMORY_USER_ID` | Yes | -- | Scopes all operations to this user. Server refuses to start without it. |
| `DATABASE_URL` | Yes | `postgresql://localhost:5432/memory_sdk` | PostgreSQL connection string. |
| `PYDANTIC_AI_MODEL` | No | -- | pydantic-ai model string in `provider:model` format (e.g., `ollama:ministral-3`, `anthropic:claude-haiku-4-5-20251001`). |
| `ANTHROPIC_API_KEY` | For Anthropic models | -- | Anthropic API key. Read by pydantic-ai's Anthropic backend. |
| `OPENAI_API_KEY` | For OpenAI models | -- | OpenAI API key. Read by pydantic-ai's OpenAI backend. |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama API endpoint. |
| `MEMORY_EXTRACTION_MAX_TOKENS` | No | `8192` | Max tokens for LLM extraction. Reasoning models consume thinking tokens before output; 8192 covers most cases. |
| `MEMORY_CONFIG` | No | -- | Path to custom TOML config file. |
| `MEMORY_EXTRACTION_TEMPLATE_PATH` | No | -- | Path to override extraction prompt (markdown with header schema). |
| `HF_HUB_OFFLINE` | No | -- | Set to `1` to skip HuggingFace HTTP checks on startup. Use after the embedding model has been cached locally. |
| `SERVER_HOST` | No | `localhost` | Server bind host (streamable-http transport). |
| `SERVER_PORT` | No | `8000` | Server bind port (streamable-http transport). |
| `MEMORY_RECALL_TOKENS_ENABLED` | No | `true` | Enable recall token disambiguation. |
| `MEMORY_RECALL_TOKENS_TOP_K` | No | `5` | Max related traces for token assessment. |
| `MEMORY_RECALL_TOKENS_THRESHOLD` | No | `0.42` | Min cosine similarity for related trace lookup at write time. |
| `MEMORY_RECALL_TOKENS_STRENGTH_THRESHOLD` | No | `0.1` | Min token strength to activate. |
| `MEMORY_RECALL_TOKENS_REINFORCE_BOOST` | No | `0.1` | Strength increment on activation. |
| `MEMORY_RECALL_TOKENS_DECAY_FACTOR` | No | `0.9` | Inactive token decay per consolidation. |
| `MEMORY_RECALL_TOKENS_HOP_DECAY` | No | `0.85` | Signal attenuation per token hop during propagation. |
| `MEMORY_RECALL_TOKENS_PROPAGATION_BLEND` | No | `0.5` | Weight of propagated signal in the additive blend. |
| `MEMORY_RECALL_TOKENS_MAX_ROUNDS` | No | `3` | Max re-seeding iterations at query time. |
| `MEMORY_RECALL_TOKENS_STABILITY_THRESHOLD` | No | `0.95` | Top-K overlap fraction to stop re-seeding early. |
| `MEMORY_RECALL_TOKENS_TOP_SEEDS` | No | `3` | Token-discovered traces used as seeds per re-seeding round. |
| `MEMORY_RECALL_TOKENS_SYSTEM_PROMPT` | No | -- | Override situational-assessment system prompt (inline string). |
| `MEMORY_RECALL_TOKENS_USER_PROMPT` | No | -- | Override situational-assessment user prompt (inline string). |

## Provider

| `PYDANTIC_AI_MODEL` prefix | Required credential |
|----------------------------|---------------------|
| `anthropic:...` | `ANTHROPIC_API_KEY` |
| `openai:...` | `OPENAI_API_KEY` |
| `openrouter:...` | `OPENROUTER_API_KEY` (e.g. `openrouter:google/gemini-3-flash-preview`) |
| `ollama:...` | `OLLAMA_BASE_URL` (defaults to `http://localhost:11434/v1`) |

Reasoning models (Qwen3, DeepSeek-R1) consume thinking tokens from the extraction budget. If `remember` returns extraction errors, increase `MEMORY_EXTRACTION_MAX_TOKENS` or set `MEMORY_CONFIG` to a custom TOML file with `[extraction] max_tokens = 8192`.

## Requirements

- Python 3.12+
- PostgreSQL 17 with pgvector

## License

MIT
