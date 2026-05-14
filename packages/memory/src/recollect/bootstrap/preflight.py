"""Bootstrap preflight: invariant checks runnable independently from apply().

Currently a single check (embedding_contract_matches). The InvariantCheck
shape is open for additional checks as drift classes surface — extension
point, not speculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from recollect.exceptions import BootstrapError
from recollect.storage_ops import get_embedding_contract

if TYPE_CHECKING:
    import asyncpg

    from recollect.embeddings import FastEmbedProvider


@dataclass(frozen=True)
class InvariantCheck:
    name: str
    ok: bool
    detail: str


def preflight_embedding_contract(
    *,
    stored: tuple[str, str] | None,
    current: tuple[str, str],
) -> InvariantCheck:
    if stored is None:
        return InvariantCheck(
            name="embedding_contract_matches",
            ok=True,
            detail="No stored contract; m003 will record current on next apply.",
        )
    if stored[0] != current[0]:
        return InvariantCheck(
            name="embedding_contract_matches",
            ok=False,
            detail=f"stored model={stored[0]!r} != current model={current[0]!r}",
        )
    if stored[1] != current[1]:
        return InvariantCheck(
            name="embedding_contract_matches",
            ok=False,
            detail=(
                f"stored task_prefix_version={stored[1]!r} != "
                f"current={current[1]!r}"
            ),
        )
    return InvariantCheck(
        name="embedding_contract_matches",
        ok=True,
        detail=f"{current[0]} ({current[1]})",
    )


def raise_on_failures(results: list[InvariantCheck]) -> None:
    failures = [r for r in results if not r.ok]
    if not failures:
        return
    detail = "; ".join(f"{r.name}: {r.detail}" for r in failures)
    raise BootstrapError(f"Preflight invariants failed: {detail}")


async def preflight_all(
    pool: asyncpg.Pool[asyncpg.Record],
    provider: FastEmbedProvider,
) -> list[InvariantCheck]:
    """Run all invariant checks against a connected DB + running provider."""
    stored = await get_embedding_contract(pool)
    return [
        preflight_embedding_contract(stored=stored, current=provider.contract()),
    ]
