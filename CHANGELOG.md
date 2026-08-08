# Changelog

## [v0.16.0] - 2026-08-08

### Added
- `recall_tokens.write_time_anchor_k` (default 5; 0 disables): entity-edge neighbors join write-time token assessment below the similarity threshold -- semantically opaque chain tails reach the assessor via shared-entity evidence

### Changed
- every auxiliary ranking signal is relevance-gated: salience boosts (`significance`, `|valence|`) multiply by `max(effective_sim, 0)` in fused scoring; `_compute_fact_relevance` is `max(sim, 0) * (0.7 + 0.3*confidence)` -- embedding-less facts rank last; token propagation stays ungated (rescue tier)

### Fixed
- `revise` assessment on an archived recall token reactivates it; an archived token updated in place no longer stays dormant
- extraction-path fact dedup user-scoped: a restated fact matches only the writer's own rows -- no cross-user mention crediting, promotion, or write swallowing; supersession inherits the scoped read
- shared-key association reads user-scoped through `memory_traces`: the `max_links` window is per-user, so a trace links to its own entity/concept siblings instead of earlier users' rows absorbing the cap

## [v0.15.0] - 2026-08-06

### Added
- `retrieval.trace_similarity_threshold` (default 0.0 = off): absolute similarity floor on storage-path trace search; `<= 0` preserves prior behavior exactly

### Changed
- `recollect-mcp` pins `mcp>=1.26.0,<2` (mcp 2.0.0 removed `mcp.server.fastmcp`); `serverInfo` advertises package name and version

### Fixed
- recall safety net re-arms per server session; a skip-reflect session's first recall surfaces pinned + health/dietary facts

## [v0.14.1] - 2026-06-19

### Added
- `packages/memory/prompts/extraction.md`: built-in extraction system prompt exported as customization reference; copy and pass via `--extraction-prompt`
- `packages/memory/prompts/token-assessment.md`: built-in token-assessment prompts (two-section: system `---` user template) exported as customization reference; override via `--token-prompt`

## [v0.14.0] - 2026-06-14

### Added
- `PREDICATE_CARDINALITY` table in `llm/types.py`: `set | current | functional` per predicate; missing-key default `set` (additive -- unclassified predicates accumulate, no data loss)
- `_find_contradicting_fact` returns `None` for `set`-cardinality predicates: a different object on a `set` predicate is an addition, not a contradiction

## [v0.13.0] - 2026-06-14

### Changed
- `pin` (`recollect-mcp`) returns formatted text; persona-fact embeddings no longer serialize into the tool result (was `list[PersonaFact]`). SDK `pin() -> list[PersonaFact]` contract unchanged

### Fixed
- `pin(trace_id)` reconciles per relation against extraction-time facts: a live non-archived subject+predicate+object twin flips to `pinned` in place, only an unmatched relation inserts -- first pin of an extraction-backed trace no longer duplicates the candidate fact or its concept embeddings
- `pin` reconcile read scoped to the trace's `user_id`; the `subject="user"` match no longer crosses users
- re-pin over an archived twin inserts a fresh `pinned` row and retains the archived row

## [v0.12.0] - 2026-06-14

### Added
- `persona.recall_relevance_floor` (0.65): absolute blended-`S` floor on non-safety persona facts at recall; `{health,dietary}` categories and pinned-when-ranked bypass it
- `persona.bridge_activation_floor` (0.0 = OFF): situational-grounding bridge recovers a below-floor persona fact whose `source_trace` is token-activated at `propagated_sim >= floor`; `<= 0` disables

### Changed
- `recall` (`recollect-mcp`) surfaces persona facts only through the recall-floored `think_about` path; the first unreflected call surfaces a safety net (pinned + health/dietary) instead of the full persona graph -- full relational context is `reflect` / the `primer` resource

## [v0.11.0] - 2026-06-12

### Added
- `MemoryTrace.status` extended with `"forgotten"`: explicit user retraction; never auto-revives (unlike `"archived"`, which revives on relevance)
- `forget_trace(trace_id)`: `UPDATE ... SET status='forgotten' WHERE status='active'`; `reactivate_trace` (WHERE status='archived') never touches forgotten rows
- `_gate_retired_candidates`: single lifecycle gate on merged candidates -- `forgotten` dropped regardless of score, `archived` revive above `reactivation_floor`

### Changed
- `forget()` flips the trace to `'forgotten'` (explicit retraction), not `'archived'` (natural fade); semantics surfaced in MCP tool description

## [v0.10.0] - 2026-06-12

### Added
- Archive lifecycle: `forget()` archives trace + derived facts; safety-critical (health/dietary/constraint) and pinned facts retained unless `force=True`; returns `ForgetResult` with per-fact dispositions
- `erase(trace_id)`: hard-delete escape hatch; normal path is `forget()`
- Relevance-gated reactivation: archived traces surfaced above `retrieval.reactivation_floor` revive automatically mid-query
- Telescoping decay anchor (`memory_traces.last_decayed_at`): per-pass factors compose; grace windowed from `recency_anchor` (not creation date)
- `MemoryTrace.status` / `FactStatus += "archived"` / `m005_trace_status` migration
- `m006_decay_anchor` migration: `last_decayed_at TIMESTAMPTZ NULL`
- `pin(trace_id)` returns `list[PersonaFact]` (promotes extracted relations; generic SPO fallback on empty extraction)
- `unpin(fact_id)` archives (was: demote to promoted)
- User isolation: `think_about(user_id=)` scopes all retrieval channels; `facts(user_id=)` kwarg; MCP reflect/primer/facts pass `app.user_id`
- Contribution-gated token reinforcement: only tokens that propagated at least one non-seed trace are reinforced
- Token decay window: `recall_tokens.decay_inactivity_seconds = 1800`; recently active tokens skip decay

### Changed
- `TraceStore.apply_strength_factor(factor)` replaces `update_trace_strength(new_strength)`: SQL-side atomic multiply
- `PoolManager.get_pool()` single-flight via `asyncio.Lock()`
- Config layered deep-merge: packaged `config.toml` always loads; user file partial-overrides; `_load_defaults` is bootstrap floor only
- Monotonic blend: `effective_sim = max(base, blend)`; weak concepts never penalize an otherwise-strong match
- Unified fact-ordering predicate: `_compute_fact_relevance` unconditional; selection and assembly share one predicate
- Supersede inherits surfacing status via rank (`pinned:2 > promoted:1 > archived/candidate:0`)
- `FastEmbedProvider.from_config(config)` single construction path; `m003` stamp derives the same provider as `CognitiveMemory.__init__`

### Fixed
- Write-path: `experience()` raises `ValueError` when extractor wired + `auto_extract` on + `user_id=None`
- Write-path: `store_trace` precedes `buffer.add`; storage failure cannot strand a trace in working memory
- `_embed_fact_tags` gated on actually-stored fact; duplicate-path writes produce zero orphan embeddings; `m004_fact_orphan_cleanup` removes the historical backlog
- `erase()` deletes orphaned concept embeddings (`concept_embeddings` has no FK on its polymorphic owner)

## [v0.8.1] - 2026-05-19

### Changed
- `FastEmbedProvider(cache_dir=...)` constructor parameter; default `~/.cache/recollect/fastembed`, replacing fastembed's tempdir-based default
- `embedding.cache_dir` config key threaded through `MemoryConfig` to `CognitiveMemory`'s default provider
- `recollect-mcp` deps: `recollect>=0.8.1` floor; `uvx` resolution pinned above pre-task-prefix (-Q) builds
