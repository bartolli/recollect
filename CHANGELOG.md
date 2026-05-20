# Changelog

## [v0.8.1] - 2026-05-19

### Changed
- `FastEmbedProvider(cache_dir=...)` constructor parameter; default `~/.cache/recollect/fastembed`, replacing fastembed's tempdir-based default
- `embedding.cache_dir` config key threaded through `MemoryConfig` to `CognitiveMemory`'s default provider
- `recollect-mcp` deps: `recollect>=0.8.1` floor; `uvx` resolution pinned above pre-task-prefix (-Q) builds
