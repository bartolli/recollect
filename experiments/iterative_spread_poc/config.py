import os
from dataclasses import dataclass


@dataclass
class PocConfig:
    database_url: str = os.environ.get("DATABASE_URL", "postgresql://localhost/memory_v3")
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"
    embedding_dimensions: int = 768
    # Spreading activation parameters
    spread_decay: float = 0.7  # Activation decay per hop
    spread_threshold: float = 0.1  # Stop spreading below this level
    spread_max_depth: int = 2  # Max hops for fixed CTE
    spread_weight: float = 0.1  # Score bonus = activation * spread_weight
    # Iterative parameters
    iter_max_rounds: int = 3  # Max re-seeding rounds
    iter_top_seeds: int = 3  # How many top results become seeds each round
    iter_stability_threshold: float = 0.95  # Stop if top-K overlap exceeds this
    # Retrieval
    recall_top_k: int = 10
    # No LLM needed -- associations are explicit in scenario
