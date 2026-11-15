# Changelog

## [v0.21.0] - 2026-11-15

### Added
- correction: `CognitiveMemory.experience(content, supersedes=[trace ids])` and MCP `remember(content, supersedes=[ids])`. Every id is validated before extraction: a missing trace, another user's trace or a forgotten trace raises `TraceNotFoundError` with no model call and no write. Then the replacement is stored, each original is forgotten through `forget`, and the replacement's persona facts are derived
- the replacement stands where the original stood: when an original carried a live persona fact, the replacement's relations pass the fact gate whatever its `fact_type`, and a derived fact with an archived original's subject and predicate inherits that status; a pinned or safety fact of an original is retained, and restated with the same triple it takes the replacement's content, embedding, tags and source trace once, with no mention credit
- `CognitiveMemory.experience_result` -> `ExperienceResult`: `trace`, `superseded` (one `ForgetResult` per original), `unsuperseded` (originals whose forget hit a storage failure; the others are still attempted) and `unstored_facts` (derived persona facts whose store write failed); `experience` returns the trace
- `CognitiveMemory.think_about_window` -> `ThoughtWindow`: the thoughts and `withheld`, the items the budget kept out; `render` applies the budget to the token estimate of the rendered text
- `CognitiveMemory.surface_relevant_facts_result` -> `SurfacedFacts` (`facts`, `lookup_failed`): a failed non-safety lookup keeps the safety facts and sets the flag; `surface_relevant_facts` returns the facts alone
- `ExperienceResult`, `SurfacedFacts` and `ThoughtWindow` are exported from `recollect`
- MCP `remember` prints after its confirmation each superseded memory with its archived fact count, each retained fact with its id, each memory not superseded with the instruction to run `forget` on it, and each persona fact that was not stored; `supersedes` takes the 8-character id or the full one
- server instructions and the `remember` tool description: one memory holds one fact or one situation useful on its own, with the context needed to find it; unrelated topics split into separate memories

### Changed
- BREAKING: `forget` on an already-forgotten trace reruns the fact pass under the guard (was: `TraceNotFoundError`): ordinary facts are archived, pinned and safety facts are retained, `force` archives those too; a missing id stays not-found; MCP `forget` on an already-forgotten memory answers with its counts
- BREAKING: `pin` pins the relations at or above `persona.confidence_threshold`, the gate extraction applies (was: every extracted relation); when none passes, the trace content is pinned as one `user noted` fact, and a second `pin` of the trace reuses that row
- BREAKING: `think_about` raises `StorageError` when the persona fact lookup fails (was: logged, thoughts returned without persona facts); MCP `recall` replies a fixed storage-failure notice and keeps the safety net pending
- BREAKING: the `FactStore` protocol gains `adopt_statement(fact_id, statement)`; a custom fact store implements it
- a restated persona fact adopts the newest statement: when another trace restates its triple, the row takes that statement's content, embedding, context tags and source trace; the mention count and promotion at the second mention are unchanged; `pin` adopts on the same rule, and a forgotten trace never rewrites a live row; a triple repeated inside one trace keeps its first context
- a fact's source is its latest stating trace: forgetting an earlier stating trace leaves the fact, forgetting the latest archives it
- `think_about` assembles persona facts and traces together: an item enters only when the complete response with it fits `token_budget`; one that does not fit is skipped, never truncated; retrieval credit and revival follow the returned window
- MCP `recall`: the printed text, safety net included, is within every accepted `token_budget`; the minimum is 23; a budget under it and an empty query are tool errors raised before the safety net is claimed; a safety net that does not fit stays pending behind a notice; matched memories that do not fit read as budget exhaustion, not as no match
- after the trace is stored, a failed step is reported or logged and the write continues: a displacement decay write, a concept, fact-tag or persona-fact embedding (the fact is stored without a vector), a supersession, a persona fact store write; MCP `remember` confirms the stored memory and says what did not complete (was: a failure reply for a memory it had stored)
- MCP `remember`: a failed fact surfacing after the write replies with the confirmation and says the related persona facts could not be loaded; a partial lookup prints the loaded facts and says the others could not be loaded
- `entity_relations.context` holds the extractor's relation context verbatim, with the literal triple as the fallback for a blank context; rows written earlier keep the literal triple
- dev gate tools are capped at the locked minor: `mypy>=1.19.1,<1.20`, `ruff>=0.15.5,<0.16`
- version lockstep 0.21.0; `recollect-mcp` requires `recollect>=0.21.0`

### Fixed
- `PydanticAIProvider` retries a rejected `temperature` by the settings its own request sent: concurrent requests on a cold provider each retry without it, and the rejection is logged once per provider

### Removed
- BREAKING: `mark_activated` is removed from the trace store, the storage protocol and `PostgresStorage`; `activation_boost_factor`, `apply_activation_boost` and the `retrieval.activation_boost` key are removed

## [v0.20.0] - 2026-11-13

### Added
- entity registry: `entities` holds a per-user identity and type, `entity_names` every surface form, unique per `(name_key, user_id)`; `name_key` is a generated column (whitespace collapsed, trimmed, lowercased); every mention resolves to its entity at write
- `CognitiveMemory.entities(user_id=None, limit=50)`: `KnownEntity` list, most-mentioned first, with names (oldest first), type, and trace count
- `CognitiveMemory.merge_entities(source_id, into_id)`: names, mentions, and relation endpoints move to the target and the source is deleted; same-owner pairs only; raises `EntityNotFoundError` or `ValueError`; nothing merges automatically
- `CognitiveMemory.entity_edges(entity_id, user_id=None, limit=50)`: `EntityEdge` per source, relation, and target, with the greatest weight and the distinct-trace count; assertions of forgotten traces are hidden
- `KnownEntity` and `EntityEdge` are exported from `recollect`; `EntityNotFoundError` is in `recollect.exceptions`
- migrations `m007_entity_registry` and `m008_relation_assertions` apply at first connect; `m007` backfills `entity_id` on every mention written before it, so an upgraded database keeps entity associations between new and older traces

### Changed
- `retrieval.selection_threshold` default: 0 (was 0.1). Semantic search admits a trace at any strength; an archived trace is a candidate, revives when its fused score reaches `retrieval.reactivation_floor`, and is delivered unrevived below it; forgotten traces stay excluded. A value above 0 restores the strength gate
- entity linking matches through entity identity: surface forms that differ by case or whitespace link to one entity
- `entity_relations` holds one row per asserting trace, unique on `(source_trace_id, source_entity, relation, target_entity)`; `erase()` removes a trace's assertions, and a forgotten trace's assertions are hidden at read
- `erase()` deletes a name once no remaining mention uses its key, and an entity once it has no mention
- a relation asserted before `m008` keeps NULL endpoint ids: `entity_edges` and `merge_entities` do not see it
- a 0.19.x process must stop before a 0.20.0 process first connects to the same database: its relation upsert names an index `m008` drops, and a mention it writes after `m007` is not backfilled
- README and config comments describe the fused score as bi-encoder similarity lifted where the concept blend exceeds it; `extraction.embed_relation_tags` stays off
- version lockstep 0.20.0; `recollect-mcp` requires `recollect>=0.20.0`

### Removed
- BREAKING: `get_relations` and `get_related_entities` are removed from the relation store, the storage protocol, and `PostgresStorage`

## [v0.19.0] - 2026-11-12

### Added
- `CognitiveMemory.traces_by_id_prefix` / `facts_by_id_prefix`: traces or persona facts whose id starts with a prefix, every status, scoped by `user_id`

### Changed
- MCP printers emit the first 8 characters of an id: `recall` on each related-memory line (printed no id), and `remember`, `reflect`, `memory://facts`, and the `pin` result (printed full ids); persona-fact lines in `recall` carry no id
- `forget`, `pin`, and `unpin` accept the 8-character id or the full id and resolve it within the server's user: one match resolves; several are refused with the candidates' full ids and nothing is written; none is not-found; an id under 8 characters is refused before any lookup
- a full id that belongs to another user no longer resolves through `forget`, `pin`, or `unpin`
- `forget` and `unpin` replies carry the full id and the opening words or triple of what they acted on
- `pin` returns not-found and ambiguity for an unresolved id as tool text (an unknown id raised a tool error)
- SDK `forget`, `pin`, and `unpin` keep exact-id contracts: resolution lives in the MCP server
- version lockstep 0.19.0; `recollect-mcp` requires `recollect>=0.19.0`

### Fixed
- the `m002_user_id_backfill` refusal names `MEMORY_USER_ID`, the env var mapped to `server.user_id` (it named `MEMORY_SERVER_USER_ID`, which is never read)

## [v0.18.0] - 2026-11-12

### Changed
- `recollect-mcp` requires `mcp>=2.3.0,<3` and builds on `mcp.server.mcpserver.MCPServer` (was `mcp>=1.26.0,<2` on `FastMCP`); environments that pin `mcp<2` cannot co-install it
- the server answers both the `initialize` handshake (protocol 2025-11-25) and the `server/discover` handshake (protocol 2026-07-28), on stdio and on streamable-http
- streamable-http: the lifespan runs once per server process (mcp 1 ran it once per HTTP session) -- one `CognitiveMemory`, connection pool, consolidation worker, working memory, and recollect session serve every HTTP client
- the skip-reflect safety net is one-shot per connection: the stdio process, or the streamable-http `Mcp-Session-Id`; a request that carries no session id is never primed, so it gets the net on every recall and `reflect` does not suppress it
- `server.host` and `server.port` are passed at `run()`; `serverInfo.version` comes from the `MCPServer` `version` argument
- version lockstep 0.18.0; `recollect-mcp` requires `recollect>=0.18.0`; the SDK package carries no code change

## [v0.17.1] - 2026-11-12

### Added
- `CognitiveMemory.health_report()` -> `HealthReport`: trace, fact, and session counts by status, degraded traces (active, no extracted concepts), surfacing facts total vs reachable, propagating vs dormant recall tokens, and violated invariants; unreachable storage yields `status=degraded` instead of raising; token counts are store-wide
- `CognitiveMemory.facts` / `get_persona_facts`: `statuses` filters in SQL ahead of the limit; `limit=None` reads unbounded; the default stays a 50-row browse page

### Changed
- `memory://health` payload is `HealthReport` (was a bare `status`)
- `forget` accepts archived traces (was active only)
- `recollect` requires `pydantic-ai>=2.67` -- the release-verified resolved closure; `recollect-mcp` requires `recollect>=0.17.1`

### Fixed
- promoted and pinned persona facts reach reflect, `memory://primer`, `memory://facts`, the skip-reflect safety net, write-time safety surfacing, and the recall promoted supplement regardless of how many newer facts exist: every surfacing read is status-filtered and unbounded (was the 50 newest facts, status-filtered after the cut)
- persona-fact dedup and pin reconciliation read every live fact for the subject: a restated fact outside the 50 newest increments its mention count instead of writing a duplicate row
- forgotten traces are excluded from semantic search in SQL ahead of the limit, and from write-time assessment candidates
- `PydanticAIProvider` serves models that refuse `temperature`: a refusal that names it, as an HTTP 400 or as `ModelRequestRejectedError`, is retried without it and `temperature` is omitted for the provider's lifetime; every `ModelAPIError` translates to `ExtractionError` (0.17.0 always sent `temperature` and stored the trace with empty extraction on a refusal)

## [v0.17.0] - 2026-08-09

### Added
- `recall_tokens.witness_bound_margin` (default 0.0 = hard bound ON; negative disables): headroom over the vouching seed's evidence for propagated scores

### Changed
- token propagation is witness-bounded: a propagation-boosted candidate's fused score caps at its best vouching seed's evidence plus the margin, and never falls below the candidate's own un-propagated score -- boosted group mass no longer displaces answer-bearing traces from top-K selection

## [v0.16.1] - 2026-08-08

### Changed
- `recollect` requires `pydantic-ai>=2.27` -- the floor is the release-verified resolved closure; `recollect-mcp` carries no pydantic-ai pin of its own and inherits it via `recollect>=0.16.1`

### Fixed
- MCP fact surfaces (reflect, `memory://facts`, pin result, remember context block) emit full fact ids; `unpin` matches ids exactly, so pin -> unpin round-trips on tool output alone

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
- recall safety net re-arms when delivery fails: a recall that errors before delivery leaves the one-shot net armed; it fires once per delivered first recall (pinned + health/dietary facts on a skip-reflect session)

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
