"""Configuration management for Memory SDK.

Layered resolution, later layers deep-merge over earlier ones:

1. Bootstrap floor (`_load_defaults`) -- only the keys the convenience
   properties read without a fallback; survives a broken wheel.
2. Packaged config.toml -- the canonical cognitive-parameter defaults;
   always loads.
3. First existing user TOML (constructor path > MEMORY_CONFIG > cwd
   memory.toml) -- a partial override never drops packaged values.
4. Env overrides (`_apply_env_overrides`).

Supports dot-notation access (e.g., config.get('memory.decay_rate')).
"""

import os
import tomllib
from pathlib import Path
from typing import Any


class MemoryConfig:
    """Configuration with cognitive model defaults."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config = self._load_defaults()

        packaged = Path(__file__).parent / "config.toml"
        if packaged.exists():
            self._load_from_file(packaged)

        for path in self._get_config_paths(config_path):
            if path.exists():
                self._load_from_file(path)
                break

        self._apply_env_overrides()

    def _get_config_paths(self, custom_path: Path | None) -> list[Path]:
        """User-supplied config candidates; first existing file wins.

        The packaged config.toml is not a candidate -- it always loads
        before this layer, so a user file overrides only the keys it sets.
        """
        paths: list[Path] = []
        if custom_path:
            paths.append(custom_path)
        if env_path := os.getenv("MEMORY_CONFIG"):
            paths.append(Path(env_path))
        paths.append(Path.cwd() / "memory.toml")
        return paths

    def _load_defaults(self) -> dict[str, Any]:
        """Bootstrap floor: only the key paths the convenience properties
        read without a fallback. Cognitive-parameter defaults live in the
        packaged config.toml, which always merges over this -- the drift
        test pins every value here to its packaged twin.
        """
        return {
            "database": {
                "url": "postgresql://localhost:5432/memory_sdk",
            },
            "memory": {
                "consolidation_threshold": 0.5,
            },
            "working_memory": {
                "capacity": 7,
            },
            "embedding": {
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "dimensions": 768,
            },
        }

    def _load_from_file(self, path: Path) -> None:
        with open(path, "rb") as f:
            file_config = tomllib.load(f)
            self._deep_merge(self._config, file_config)

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides (highest priority).

        Resolution order: defaults -> TOML -> env vars.
        Only set values are applied; unset env vars leave TOML/defaults intact.
        """
        env_map: list[tuple[str, str]] = [
            ("DATABASE_URL", "database.url"),
            ("PYDANTIC_AI_MODEL", "extraction.pydantic_ai_model"),
            ("MEMORY_EXTRACTION_MAX_TOKENS", "extraction.max_tokens"),
            ("MEMORY_EXTRACTION_INSTRUCTIONS", "extraction.instructions"),
            ("MEMORY_EXTRACTION_TEMPLATE_PATH", "extraction.template_path"),
            ("SERVER_HOST", "server.host"),
            ("SERVER_PORT", "server.port"),
            ("MEMORY_USER_ID", "server.user_id"),
            ("MEMORY_RECALL_TOKENS_ENABLED", "recall_tokens.enabled"),
            ("MEMORY_RECALL_TOKENS_TOP_K", "recall_tokens.write_time_top_k"),
            ("MEMORY_RECALL_TOKENS_THRESHOLD", "recall_tokens.write_time_threshold"),
            (
                "MEMORY_RECALL_TOKENS_STRENGTH_THRESHOLD",
                "recall_tokens.strength_threshold",
            ),
            ("MEMORY_RECALL_TOKENS_REINFORCE_BOOST", "recall_tokens.reinforce_boost"),
            ("MEMORY_RECALL_TOKENS_DECAY_FACTOR", "recall_tokens.decay_factor"),
            ("MEMORY_RECALL_TOKENS_HOP_DECAY", "recall_tokens.hop_decay"),
            (
                "MEMORY_RECALL_TOKENS_PROPAGATION_BLEND",
                "recall_tokens.propagation_blend",
            ),
            ("MEMORY_RECALL_TOKENS_MAX_ROUNDS", "recall_tokens.iter_max_rounds"),
            (
                "MEMORY_RECALL_TOKENS_STABILITY_THRESHOLD",
                "recall_tokens.iter_stability_threshold",
            ),
            ("MEMORY_RECALL_TOKENS_TOP_SEEDS", "recall_tokens.iter_top_seeds"),
            (
                "MEMORY_RECALL_TOKENS_SYSTEM_PROMPT",
                "recall_tokens.assessment_system_prompt",
            ),
            (
                "MEMORY_RECALL_TOKENS_USER_PROMPT",
                "recall_tokens.assessment_user_prompt",
            ),
        ]

        for env_var, config_path in env_map:
            value = os.getenv(env_var)
            if value is not None:
                self._set(config_path, value)

    def _set(self, path: str, value: Any) -> None:
        """Set a config value by dot notation."""
        keys = path.split(".")
        target = self._config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        # Preserve type for numeric config values
        existing = target.get(keys[-1])
        if isinstance(existing, bool):
            target[keys[-1]] = str(value).lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            target[keys[-1]] = int(value)
        elif isinstance(existing, float):
            target[keys[-1]] = float(value)
        else:
            target[keys[-1]] = value

    # -- Convenience properties --

    @property
    def database_url(self) -> str:
        return str(self._config["database"]["url"])

    @property
    def working_memory_capacity(self) -> int:
        return int(self._config["working_memory"]["capacity"])

    @property
    def consolidation_threshold(self) -> float:
        return float(self._config["memory"]["consolidation_threshold"])

    @property
    def embedding_dimensions(self) -> int:
        return int(self._config["embedding"]["dimensions"])

    @property
    def extraction_instructions(self) -> str:
        """Get extraction instructions."""
        return str(self.get("extraction.instructions", ""))

    @property
    def recall_token_system_prompt(self) -> str:
        """Get recall token assessment system prompt override."""
        return str(self.get("recall_tokens.assessment_system_prompt", ""))

    @property
    def recall_token_user_prompt(self) -> str:
        """Get recall token assessment user prompt override."""
        return str(self.get("recall_tokens.assessment_user_prompt", ""))

    @property
    def server_user_id(self) -> str:
        """Get server user ID."""
        return str(self.get("server.user_id", ""))

    def get(self, path: str, default: Any = None) -> Any:
        """Get config value by dot notation (e.g., 'memory.decay_rate')."""
        keys = path.split(".")
        value: Any = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __repr__(self) -> str:
        return (
            f"MemoryConfig(wm={self.working_memory_capacity}, "
            f"consolidation={self.consolidation_threshold})"
        )


# Global config instance
config = MemoryConfig()
