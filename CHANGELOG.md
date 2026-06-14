# Changelog

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
