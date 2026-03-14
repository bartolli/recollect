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
| `recall` | `query: str` | Retrieve relevant memories. Returns persona facts as context followed by matching traces. |
| `reflect` | -- | Load persona context for the current session. Call before responding to any user message. |
| `pin` | `content: str` | Promote a statement to a permanent persona fact. |
| `unpin` | `fact_id: str` | Remove a persona fact. |
| `forget` | `trace_id: str` | Delete a memory trace. |

## Resources

| URI | Description |
|-----|-------------|
| `memory://primer` | Relational graph of persona facts. Read at conversation start for user context. |
| `memory://facts` | All active persona facts with confidence scores and timestamps. |
| `memory://health` | Server and database health status. |

Clients that support MCP resources get session priming automatically via `primer`. For clients that don't, `reflect` loads the same context as a tool call. If neither is invoked, the server injects the primer on the first tool call of the session.

## Client configuration

Add to `.mcp.json` (Claude Code) or `claude_desktop_config.json` (Claude Desktop):

```json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": ["recollect-mcp"],
      "env": {
        "MEMORY_USER_ID": "your-user-id",
        "DATABASE_URL": "postgresql://user@localhost:5432/dbname",
        "PYDANTIC_AI_MODEL": "anthropic:claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

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
| `HF_HUB_OFFLINE` | No | -- | Set to `1` to skip HuggingFace HTTP checks on startup. Use after the embedding model has been cached locally. |
| `SERVER_HOST` | No | `localhost` | Server bind host (streamable-http transport). |
| `SERVER_PORT` | No | `8000` | Server bind port (streamable-http transport). |
| `MEMORY_RECALL_TOKENS_ENABLED` | No | `true` | Enable recall token disambiguation. |
| `MEMORY_RECALL_TOKENS_TOP_K` | No | `5` | Max related traces for token assessment. |
| `MEMORY_RECALL_TOKENS_THRESHOLD` | No | `0.42` | Min cosine similarity for related trace lookup at write time. |
| `MEMORY_RECALL_TOKENS_STRENGTH_THRESHOLD` | No | `0.1` | Min token strength to activate. |
| `MEMORY_RECALL_TOKENS_SCORE_BONUS` | No | `0.1` | Gated additive bonus per token. |
| `MEMORY_RECALL_TOKENS_REINFORCE_BOOST` | No | `0.1` | Strength increment on activation. |
| `MEMORY_RECALL_TOKENS_DECAY_FACTOR` | No | `0.9` | Inactive token decay per consolidation. |

## Provider

| `PYDANTIC_AI_MODEL` prefix | Required credential |
|----------------------------|---------------------|
| `anthropic:...` | `ANTHROPIC_API_KEY` |
| `openai:...` | `OPENAI_API_KEY` |
| `ollama:...` | `OLLAMA_BASE_URL` (defaults to `http://localhost:11434/v1`) |

Reasoning models (Qwen3, DeepSeek-R1) consume thinking tokens from the extraction budget. If `remember` returns extraction errors, increase `MEMORY_EXTRACTION_MAX_TOKENS` or set `MEMORY_CONFIG` to a custom TOML file with `[extraction] max_tokens = 8192`.

## Requirements

- Python 3.12+
- PostgreSQL 17 with pgvector

## License

MIT
