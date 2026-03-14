# Recollect

Persistent memory for LLM agents.

---

Vector search finds what's similar. It doesn't find what's *related*.

```
"How are we getting Mom from the airport?"
```

Buried in the memory store:

```
"Mom's flight lands at terminal B, 3:15 Friday"
"The car is at the mechanic until Saturday -- transmission repair"
```

The flight info shares "Mom" with the query -- vector search might find it, along with every other memory that mentions Mom. The car being unavailable shares nothing: no common words, no embedding overlap. But it's the critical constraint. The connection is situational, and the only way to surface it is to link memories that have nothing to do with each other on the surface.

Recollect is an experiment in retrieval that handles this.

---

## Packages

**[`recollect`](packages/memory/)** — The SDK. Async Python API for building memory pipelines: `experience()` to store, `think_about()` to recall, `consolidate()` to maintain. Bring your own LLM via pydantic-ai (20+ providers). Use this when you're building your own agent or pipeline.

```bash
pip install recollect    # or: uv add recollect
```

**[`recollect-mcp`](packages/memory-mcp/)** — The MCP server. Six tools (`remember`, `recall`, `pin`, `unpin`, `forget`, `reflect`) and three resources (`primer`, `facts`, `health`). Drop it into Claude Desktop, Cursor, or any MCP-compatible client and get persistent memory out of the box. Clients that support MCP resources get session priming automatically via `primer`; for clients that don't, `reflect` loads the same persona context as a tool call. If the agent skips both, the server injects the primer automatically on the first tool call of the session.

```bash
pip install recollect-mcp    # or: uv add recollect-mcp
```

Works with agent frameworks like [OpenClaw](https://github.com/openclaw/openclaw), [NanoClaw](https://github.com/qwibitai/nanoclaw), or any agent that needs persistent memory across sessions.

<details>
<summary><b>Example: personal email agent</b></summary>

<br>

A cron job or MQTT listener watches your Gmail. When new mail arrives, an agent processes it: sorts by priority, flags spam and scam, identifies actionable items, and stores memories about anything worth remembering.

The SDK handles the rest. An email confirming Mom's flight gets stored. When a follow-up arrives saying the flight was rescheduled to Saturday, the token system revises the airport pickup group -- the logistics chain updates automatically. Next morning, when you ask "what's the plan for Friday?" the agent knows the pickup moved, Dave's truck is no longer needed Friday, and surfaces the updated context.

No custom retrieval logic. The situational connections are maintained by the same write-time assessment and query-time propagation that handles everything else. You build the email ingestion; Recollect builds the memory.

</details>

## Three tiers

**Context priming** — Pinned facts load at session start. No query needed. If it matters every session, it's already in context.

**Episodic retrieval** — Fused pipeline: pgvector HNSW, entity trigram matching, spreading activation, working memory scan. Scored with ColBERT-style MaxSim concept attention (0.7) blended with bi-encoder similarity (0.3). Entity bonuses are multiplicatively gated by concept similarity to prevent entity flooding. No LLM calls at read time.

**Recall tokens** — At write time, the LLM evaluates whether a new memory belongs to an existing situational group or starts a new one. Traces get stamped with shared tokens. At query time, activation propagates through those groups, pulling in traces that vector search missed.

| Tier | What it does | Needs a query? |
|---|---|:---:|
| Context priming | Always-on facts | No |
| Episodic retrieval | Standard recall | Yes |
| Recall tokens | Situational rescue | Yes |

### Scoring formula

Each memory trace is decomposed into concept embeddings at write time. At query time, scoring uses MaxSim — the maximum cosine similarity between the query and any concept vector for a given trace:

$$s_{\text{concept}} = \max_{i} \cos(q,\; c_i)$$

This blends with the standard bi-encoder score:

$$s_{\text{eff}} = 0.7 \cdot s_{\text{concept}} + 0.3 \cdot s_{\text{bienc}}$$

The full ranking score adds significance, valence, spreading activation, entity matching, and token propagation:

$$\text{score} = s_{\text{eff}} \;+\; 0.15 \cdot \sigma \;+\; 0.05 \cdot |\nu| \;+\; 0.1 \cdot a \;+\; \underbrace{0.1 \cdot e \cdot \sigma \cdot s_{\text{concept}}}_{\text{gated entity bonus}} \;+\; \underbrace{0.5 \cdot s_{\text{prop}}}_{\text{token propagation}}$$

The entity bonus is multiplicatively gated by both significance ($\sigma$) and concept similarity ($s_{\text{concept}}$). When a trace's concepts have zero overlap with the query, the entity bonus is zero regardless of name match. This prevents entity flooding: not every "Mom" trace surfaces just because the query mentions Mom.

Token propagation signal decays per hop:

$$s_{\text{prop}} = s_{\text{anchor}} \cdot 0.85^{h} \cdot \text{strength} \cdot \sigma$$

where $s_{\text{anchor}}$ is the cosine similarity of the seed trace that activated the token, and $h$ is the hop count.

All weights are configurable via TOML.

## How tokens evolve

Tokens aren't static — they have a lifecycle. A situational group gets **created** when the LLM detects a causal chain, **extended** when new information joins it, and **revised** when the situation changes.

A memory comes in:

```
"Mom's flight lands at terminal B, 3:15 Friday"
```
```
-> no group yet (standalone travel fact)
```

A related memory arrives. The LLM detects the logistics dependency:

```
"The car is at the mechanic until Saturday -- transmission repair"
```
```
-> token created
  household | airport pickup logistics | no vehicle available Friday
```

New information joins the group:

```
"Dave offered to lend us his truck Friday afternoon"
```
```
-> token extended
  household | airport pickup logistics | no vehicle available Friday, backup vehicle offered
```

Later, things change:

```
"Mechanic called -- car is ready for pickup tomorrow"
```
```
-> token revised
  household | airport pickup logistics | car available, backup no longer needed
```

Now a query like "how are we getting Mom from the airport?" activates the token, propagates through the group, and surfaces the full logistics chain -- even though "car at the mechanic" and "Dave's truck" share no words with the query.

Two tables, a strength float, and four actions: `create`, `extend`, `revise`, `none`. No ontology, no schema.

### Token reinforcement and decay

Tokens follow a Hebbian-inspired lifecycle — connections that get used get stronger, connections that don't get used fade.

**Reinforcement.** When a token participates in successful recall, its strength increments:

$$s \leftarrow \min(1.0,\; s + 0.1)$$

**Decay.** Each consolidation pass, all active tokens lose a fraction of their strength:

$$s \leftarrow s \times 0.9$$

Tokens that fall below 0.01 are **archived** — not deleted. The label, stamps, and significance are preserved, but the token stops participating in query-time activation.

**Reactivation.** When a future write-time assessment (`extend` or `revise`) references an archived token, it comes back:

$$s \leftarrow \sigma$$

Strength resets to the token's significance ($\sigma$), not a fixed value. A high-significance token (0.85) comes back strong. A low-significance one (0.3) comes back weak but viable. From there, the normal reinforcement/decay loop takes over.

The result: frequently useful situational groups strengthen over time, unused links fade to archive, and archived tokens can be resurrected without a full LLM reassessment. The causal chain survives indefinitely at negligible storage cost.

**Iterative propagation.** At query time, token activation doesn't stop at one hop. Top discovered traces become seeds for the next round, up to 3 re-seeding iterations, stopping early when top-K ranking stabilizes (>= 95% overlap between rounds).

## Status

Working. Context priming via MCP, write-time extraction and concept decomposition, fused retrieval, full token lifecycle (create, propagate, reinforce, decay, archive, reactivate).

Open: longitudinal reinforcement validation, density inflection characterization.

> Installation docs in progress. Open an issue if you want to run it.

## Stack

- PostgreSQL 17 + pgvector
- FastEmbed local embeddings (nomic-embed-text-v1.5-Q)
- pydantic-ai for LLM extraction (20+ model providers)
- Protocol-driven storage (PEP 544), TOML config
- asyncpg, async-first

## Prerequisites

```bash
# macOS
brew install postgresql@17
brew install pgvector

# Start PostgreSQL
brew services start postgresql@17

# Create database (name it whatever you want) and enable pgvector
createdb recollect
psql recollect -c "CREATE EXTENSION IF NOT EXISTS vector"
```

The SDK bootstraps its schema on first connect. Database name, credentials, and connection security (SSL, password auth, network policies) are your responsibility -- `DATABASE_URL` accepts any standard PostgreSQL connection string.

An LLM provider is required for write-time extraction. Any [pydantic-ai supported model](https://ai.pydantic.dev/models/) works -- set `PYDANTIC_AI_MODEL` and the provider's API key:

```bash
# Example: Anthropic
export PYDANTIC_AI_MODEL="anthropic:claude-haiku-4-5-20251001"
export ANTHROPIC_API_KEY="sk-ant-..."

# Example: local Ollama (no API key needed)
export PYDANTIC_AI_MODEL="ollama:ministral-3"

# Required
export DATABASE_URL="postgresql://localhost:5432/recollect"
```

## Background

Long history with ontologies and knowledge graphs — still the right tool where relationships can be formally defined. What got me thinking differently was transformer attention: when two facts are in the context window, the model connects them regardless of surface similarity. It already knows that a car in the shop means you can't drive to the airport. It just needs both facts to be there.

Recollect tries to build retrieval that works the same way, without the complexity of a full graph. Two earlier versions explored other approaches before this one. Still an experiment.