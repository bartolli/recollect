import os
from dataclasses import dataclass


@dataclass
class PocConfig:
    database_url: str = os.environ.get("DATABASE_URL", "postgresql://localhost/memory_v3")
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"
    embedding_dimensions: int = 768
    llm: str = "ollama:ministral-3"
    llm_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    llm_max_tokens: int = 1024
    write_time_top_k: int = 5
    write_time_threshold: float = 0.42
    token_strength_threshold: float = 0.1
    token_decay_factor: float = 0.95
    token_reinforce_boost: float = 0.1
    token_score_bonus: float = 0.1
    hop_decay: float = 0.85
    significance_weight: float = 0.15
    valence_weight: float = 0.05
    recall_top_k: int = 20
    prompt_path: str | None = None
    # Iterative re-seeding parameters
    iter_max_rounds: int = 3
    iter_stability_threshold: float = 0.95
    iter_top_seeds: int = 3
    # Scoring: how much propagated_sim contributes to token-activated traces.
    # 1.0 = old behaviour (max replacement); 0.5 = conservative blend.
    propagation_blend: float = 0.5
