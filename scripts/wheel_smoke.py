"""Live wheel smoke: both handshakes on stdio, session and sessionless HTTP.

Run with the interpreter of the environment under test, never the dev
venv: it reports what that environment installed. Needs DATABASE_URL on
an EMPTY scratch database, MEMORY_USER_ID, SMOKE_HTTP_PORT (a free port),
PYDANTIC_AI_MODEL and its API key. `make -f Makefile.local release-smoke`
wires all of it.

The stdio legacy arm is the full tool cycle; the other arms check the
handshake and the per-connection safety net. Exit 0 and `SMOKE OK` on pass.
"""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from importlib.metadata import version

import asyncpg
import recollect
import recollect_mcp
from mcp import Client, StdioServerParameters

LEGACY, MODERN = "2025-11-25", "2026-07-28"
HEALTH_KEYS = {
    "status",
    "traces",
    "degraded_traces",
    "facts",
    "surfacing_total",
    "surfacing_reachable",
    "tokens_propagating",
    "tokens_dormant",
    "sessions",
    "invariants",
}
SHORT_ID = r"[0-9a-f]{8}"
UNRELATED = "When is the quarterly tax filing due?"
PORT = int(os.environ["SMOKE_HTTP_PORT"])
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


async def tool(c: Client, arm: str, name: str, **args: object) -> str:
    result = await c.call_tool(name, args)
    text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
    check(f"{arm}: tool {name} returns without error", not result.is_error, text[:90])
    return text


async def resource(c: Client, uri: str) -> str:
    return (await c.read_resource(uri)).contents[0].text


async def health(c: Client, arm: str) -> dict[str, object]:
    report = json.loads(await resource(c, "memory://health"))
    check(f"{arm}: health carries every HealthReport key", set(report) == HEALTH_KEYS)
    return report


def net_fired(recall_text: str) -> bool:
    """Gate-surfaced facts carry a score prefix; safety-net lines do not."""
    head = recall_text.split("Related memories:")[0].splitlines()
    return any(
        ln.startswith("- ") and not ln.startswith("- [score:") and "net-marker" in ln
        for ln in head
    )


async def seed_net_only_facts() -> None:
    """Pinned facts beyond the recall gate's per-query cap: the gate surfaces at
    most `persona.max_facts_per_query`, the rest reach recall by the net alone."""
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        for n in range(8):
            await conn.execute(
                "INSERT INTO persona_facts (id, subject, predicate, object, category,"
                " content, confidence, status, user_id) VALUES ($1, 'Ines Marchetti',"
                " 'prefers', $2, 'preference', $3, 0.9, 'pinned', $4)",
                str(uuid.uuid4()),
                f"net-marker item {n}",
                f"Ines Marchetti prefers net-marker item {n}",
                os.environ["MEMORY_USER_ID"],
            )
    finally:
        await conn.close()


async def stdio_legacy_full_cycle(params: StdioServerParameters) -> None:
    arm = "stdio legacy"
    async with Client(params, mode="legacy") as c:
        check(
            f"{arm}: negotiates {LEGACY}",
            c.protocol_version == LEGACY,
            str(c.protocol_version),
        )
        info = c.server_info
        check(
            f"{arm}: serverInfo is this package",
            (info.name, info.version) == ("recollect-mcp", version("recollect-mcp")),
            f"{info.name} {info.version}",
        )
        check(f"{arm}: instructions delivered", bool(c.instructions))
        tools = sorted(t.name for t in (await c.list_tools()).tools)
        check(
            f"{arm}: six tools",
            tools == ["forget", "pin", "recall", "reflect", "remember", "unpin"],
            str(tools),
        )
        uris = sorted(str(r.uri) for r in (await c.list_resources()).resources)
        check(
            f"{arm}: three resources",
            uris == ["memory://facts", "memory://health", "memory://primer"],
            str(uris),
        )
        empty = await health(c, arm)
        check(
            f"{arm}: empty store health ok",
            empty["status"] == "ok" and empty["traces"] == {},
        )

        await tool(c, arm, "reflect")
        kept = await tool(
            c,
            arm,
            "remember",
            content="Ines Marchetti's son Paolo Marchetti has a severe peanut allergy "
            "and must never be served peanuts or peanut oil.",
        )
        m = re.search(
            rf"\[({SHORT_ID})\] \(significance [\d.]+, "
            r"(\d+) entities, (\d+) concepts\)",
            kept,
        )
        assert m is not None, kept
        check(
            f"{arm}: extraction is not empty",
            int(m.group(2)) > 0 and int(m.group(3)) > 0,
            m.group(0),
        )
        recalled = await tool(
            c, arm, "recall", query="What can Paolo Marchetti not eat?"
        )
        check(f"{arm}: recall surfaces the memory", "peanut" in recalled.lower())
        pinned = await tool(c, arm, "pin", trace_id=m.group(1))
        fact_ids = re.findall(rf"\[({SHORT_ID})\]", pinned)
        check(f"{arm}: pin returns fact ids", len(fact_ids) > 0)

        pref = await tool(
            c,
            arm,
            "remember",
            content="Ines Marchetti prefers aisle seats on long-haul flights.",
        )
        p = re.search(rf"\[({SHORT_ID})\]", pref)
        assert p is not None, pref
        await tool(c, arm, "pin", trace_id=p.group(1))

        check(
            f"{arm}: facts resource shows the pins",
            "pinned" in await resource(c, "memory://facts"),
        )
        check(
            f"{arm}: primer shows the pins",
            "Paolo" in await resource(c, "memory://primer"),
        )
        mid = await health(c, arm)
        check(
            f"{arm}: surfacing reachable equals total and is nonzero",
            mid["surfacing_total"] == mid["surfacing_reachable"] != 0,
            f"{mid['surfacing_reachable']}/{mid['surfacing_total']}",
        )
        # The fact id comes from the facts resource, not from the pin reply.
        listed_fact = re.search(
            rf"^\[({SHORT_ID})\] .*peanut",
            await resource(c, "memory://facts"),
            re.MULTILINE | re.IGNORECASE,
        )
        check(f"{arm}: facts resource prints fact ids", listed_fact is not None)
        assert listed_fact is not None
        unpinned = await tool(c, arm, "unpin", fact_id=listed_fact.group(1))
        check(f"{arm}: unpin reports the archived fact", "archived" in unpinned)
        await tool(
            c,
            arm,
            "remember",
            content="Ines Marchetti wrongly noted that her car is a blue Volvo.",
        )
        # The id comes from recall output: a trace the caller did not create.
        listed = await tool(
            c, arm, "recall", query="What car does Ines Marchetti drive?"
        )
        d = re.search(rf"^- \[({SHORT_ID})\] .*Volvo", listed, re.MULTILINE)
        check(f"{arm}: recall prints an id on the memory line", d is not None)
        assert d is not None, listed
        forgotten = await tool(c, arm, "forget", trace_id=d.group(1))
        check(
            f"{arm}: forget reports the forgotten memory",
            "forgotten" in forgotten and "Volvo" in forgotten,
            forgotten[:120],
        )
        after = await tool(
            c, arm, "recall", query="What car does Ines Marchetti drive?"
        )
        check(f"{arm}: forgotten trace stays out of recall", "Volvo" not in after)
        final = await health(c, arm)
        check(f"{arm}: no degraded traces", final["degraded_traces"] == 0)
        check(
            f"{arm}: no violated invariants",
            final["invariants"] == [],
            str(final["invariants"]),
        )
        check(
            f"{arm}: trace counts by status",
            final["traces"] == {"active": 2, "forgotten": 1},
            str(final["traces"]),
        )
        print(json.dumps(final))


async def stdio_modern(params: StdioServerParameters) -> None:
    arm = "stdio modern"
    async with Client(params) as c:
        check(
            f"{arm}: negotiates {MODERN}",
            c.protocol_version == MODERN,
            str(c.protocol_version),
        )
        check(f"{arm}: six tools", len((await c.list_tools()).tools) == 6)
        check(f"{arm}: three resources", len((await c.list_resources()).resources) == 3)
        check(
            f"{arm}: first unreflected recall gets the net",
            net_fired(await tool(c, arm, "recall", query=UNRELATED)),
        )
        check(
            f"{arm}: second recall does not",
            not net_fired(await tool(c, arm, "recall", query=UNRELATED)),
        )
        check(
            f"{arm}: reflect shows the pinned preference",
            "aisle" in (await tool(c, arm, "reflect")).lower(),
        )
        await health(c, arm)


async def http(url: str) -> None:
    async with Client(url, mode="legacy") as a:
        arm = "http session A"
        check(
            f"{arm}: negotiates {LEGACY}",
            a.protocol_version == LEGACY,
            str(a.protocol_version),
        )
        check(
            f"{arm}: first recall gets the net",
            net_fired(await tool(a, arm, "recall", query=UNRELATED)),
        )
        check(
            f"{arm}: second recall does not",
            not net_fired(await tool(a, arm, "recall", query=UNRELATED)),
        )
    async with Client(url, mode="legacy") as b:
        arm = "http session B"
        check(
            f"{arm}: first recall gets its own net",
            net_fired(await tool(b, arm, "recall", query=UNRELATED)),
        )
        await tool(b, arm, "reflect")
        check(
            f"{arm}: recall after reflect does not",
            not net_fired(await tool(b, arm, "recall", query=UNRELATED)),
        )
    async with Client(url) as m:
        arm = "http sessionless"
        check(
            f"{arm}: negotiates {MODERN}",
            m.protocol_version == MODERN,
            str(m.protocol_version),
        )
        check(f"{arm}: six tools", len((await m.list_tools()).tools) == 6)
        check(
            f"{arm}: recall gets the net",
            net_fired(await tool(m, arm, "recall", query=UNRELATED)),
        )
        await tool(m, arm, "reflect")
        check(
            f"{arm}: recall after reflect still gets the net",
            net_fired(await tool(m, arm, "recall", query=UNRELATED)),
        )
        await health(m, arm)


def wait_for_port(port: int, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


async def main() -> None:
    print(
        "recollect", version("recollect"), "from", os.path.dirname(recollect.__file__)
    )
    print(
        "recollect-mcp",
        version("recollect-mcp"),
        "from",
        os.path.dirname(recollect_mcp.__file__),
    )
    for pkg in ("mcp", "pydantic-ai", "anthropic"):
        print(pkg, version(pkg))
    print("model", os.environ.get("PYDANTIC_AI_MODEL", "<unset>"))

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "recollect_mcp.server"],
        env=dict(os.environ),
    )
    await stdio_legacy_full_cycle(params)
    await seed_net_only_facts()
    await stdio_modern(params)

    with open("http-server.log", "w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "recollect_mcp.server",
                "--transport",
                "streamable-http",
            ],
            env={**os.environ, "SERVER_HOST": "127.0.0.1", "SERVER_PORT": str(PORT)},
            stdout=log,
            stderr=log,
        )
        try:
            check("http: server listens on the configured port", wait_for_port(PORT))
            await http(f"http://127.0.0.1:{PORT}/mcp")
        finally:
            server.terminate()
            server.wait(timeout=30)
    with open("http-server.log") as log:
        starts = log.read().count("recollect-mcp server started")
    check("http: one lifespan serves every client", starts == 1, f"started x{starts}")

    print("SMOKE", "FAILED: " + ", ".join(failures) if failures else "OK")
    sys.exit(1 if failures else 0)


asyncio.run(main())
