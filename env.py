"""Tempora — HUD v6 environment for temporality-reasoning deep-research agents.

Hosts three mock tools — ``search``, ``fetch``, ``get_current_time`` — that serve
a pre-generated, temporally-mixed document bundle (see ``generator.py``). The
agent must call the tools, reason about timestamps relative to the reference
time ``T`` (only knowable via ``get_current_time``), and report what is
verifiably current — excluding expired, ended, future, and undated/unconfirmable
items, including stale "right now" aggregator lists and misleading social posts.

HUD v6 shape:
  * Custom agent tools = FastMCP ``@server.tool`` functions, started in
    ``@env.initialize`` and published via ``env.add_capability(Capability.mcp(...))``.
    The agent auto-discovers them.
  * Tasks via ``@env.template()`` async generators: ``yield`` prompt -> receive
    the agent's final answer -> ``yield`` a 0..1 reward.
  * Run: ``hud serve env.py`` to iterate; ``hud eval tasks.py claude --group N``.

Per-rollout state: each task run brings up its own env process, so the FastMCP
tools read a module-level ``STATE`` populated by the template from its args
before yielding the prompt. (If a concurrency smoke test ever shows collision,
switch ``STATE`` to a run-id-keyed dict.)

Default rendering is the generator's ``--no-prose`` template prose, so the whole
environment runs with no API key. Set ``HUD_API_KEY`` and pass ``use_llm=True``
to a template to get LLM-rewritten prose via the HUD gateway.

Scoring: the agent calls ``submit_answer(items, supporting_doc_ids)`` with the
items it asserts are current and the corpus URLs it relied on. A standalone
programmatic grader (``grader.py``) scores that against the bundle's
code-computed ground truth — no free-text parsing, so rewards are model-
independent. Free text in the agent's final message is never graded.
"""
# NOTE: do NOT add `from __future__ import annotations` here. Under it, a
# @env.template param annotation becomes a string forward-ref that crashes the
# sync/deploy manifest path (TypeAdapter on a string). Keep annotations as real
# objects. (Confirmed by the sibling deepresearch env.)
import asyncio
import hashlib
import json
import re
import socket
from typing import Any, Optional

from fastmcp import FastMCP

from hud import Environment
from hud.capabilities import Capability
from hud.graders import EvaluationResult

import generator as gen
from generator import LabelForm

from grader import grade

# --------------------------------------------------------------------------- #
# Per-rollout state (one env process per rollout)
# --------------------------------------------------------------------------- #

STATE: dict = {
    "bundle": None,        # gen.DocBundle
    "now_iso": "",
    "docs": [],            # list[gen.Doc]
    "items_by_id": {},     # id -> gen.Item
    "gold_ids": [],
    "lure_ids": [],
    "difficulty": 0.3,
    "submission": None,    # agent's submit_answer payload: {items, supporting_doc_ids}
}


def _load_bundle(bundle: gen.DocBundle) -> None:
    STATE["bundle"] = bundle
    STATE["now_iso"] = bundle.now.isoformat()
    STATE["docs"] = bundle.docs
    STATE["items_by_id"] = {it.id: it for it in bundle.items}
    STATE["submission"] = None
    STATE["gold_ids"] = list(bundle.gold_ids)
    STATE["lure_ids"] = list(bundle.lure_ids)


# --------------------------------------------------------------------------- #
# Tool logic as pure functions (testable without HUD / FastMCP / keys)
# --------------------------------------------------------------------------- #

def _token_overlap(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q:
        return 0.0
    return len(q & t) / len(q)


def _req_id(*parts: str) -> str:
    """Stable Exa-style requestId (a 32-char hex digest) from the call parts."""
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h + h[:12]  # 32 chars


def _published_date_iso(d: gen.Doc) -> Optional[str]:
    """Exa returns an ISO 8601 publishedDate when it can estimate one, else null.
    We mirror that: a real ISO date when the page states an explicit or relative
    date (both are estimable — relative is anchored to T, which is our crawl/reference
    time); null when the label is vague ('this season') or missing. This is faithful
    to Exa without bypassing the trap: undated/vague sources still force the agent
    to read the body, and item-level dates inside bodies always require reading."""
    if d.published_at is None:
        return None
    if d.published_label_form in (LabelForm.EXPLICIT, LabelForm.RELATIVE):
        # Exa uses Z-suffixed UTC; we emit a naive isoformat plus Z for shape fidelity.
        return d.published_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return None


def _author_for(d: gen.Doc) -> Optional[str]:
    """A plausible author per source type, or null (Exa returns null when unknown)."""
    if d.source_type == "social_post":
        # deterministic handle from the doc id
        h = hashlib.sha1(d.doc_id.encode("utf-8")).hexdigest()[:6]
        sub = d.source_subtype or "user"
        return f"@{sub}_{h}"
    if d.source_type == "aggregator_list":
        return "editorial@aggregator"
    return None


def _find_doc(doc_id_or_url: str) -> Optional[gen.Doc]:
    """Look up a doc by its Exa id (= url) first, then by internal doc_id."""
    for d in STATE["docs"]:
        if d.url == doc_id_or_url or d.doc_id == doc_id_or_url:
            return d
    return None


def _search(query: str, limit: int = 8) -> dict:
    """Exa /search-shaped response. Ranks the bundle's docs by the generator's
    search_rank_hint (which bakes in lure/gold placement adversity) plus a small
    relevance term. Returns metadata only — title, url, publishedDate, score,
    summary, highlights — never the full body text (the agent must call fetch,
    mirroring Exa where text requires contents.text)."""
    docs = STATE["docs"]
    if not docs:
        return {"requestId": _req_id(query), "results": [], "costDollars": {"total": 0.0}}
    scored = []
    for d in docs:
        relevance = _token_overlap(query, f"{d.title} {d.snippet}")
        score = d.search_rank_hint + 0.05 * relevance
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sc, d in scored[:limit]:
        results.append({
            "id": d.url,
            "url": d.url,
            "title": d.title,
            "publishedDate": _published_date_iso(d),
            "author": _author_for(d),
            "score": round(max(0.0, min(1.0, sc)), 4),
            "summary": d.snippet,
            "highlights": [d.snippet] if d.snippet else [],
            "subpages": [],
            "extras": {"links": []},
        })
    return {
        "requestId": _req_id(query),
        "results": results,
        "costDollars": {"total": round(0.001 * len(results), 4)},
    }


def _fetch(doc_id_or_url) -> dict:
    """Exa /contents-shaped response. Accepts a single id/url OR a list of them
    (Exa's /contents endpoint batches multiple ids in one call). Each found doc
    becomes a result with its full ``text``; unknown ids are dropped. Accepts the
    id/url returned by search (or the internal doc_id)."""
    # Normalize to a list. Tolerate: a real list/tuple, a single string, or a
    # stringified JSON list (some agents pass '["url1","url2"]' as a string).
    if isinstance(doc_id_or_url, (list, tuple)):
        ids = list(doc_id_or_url)
    elif isinstance(doc_id_or_url, str):
        s = doc_id_or_url.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                ids = list(parsed) if isinstance(parsed, list) else [s]
            except Exception:
                ids = [s]
        else:
            ids = [s]
    else:
        ids = [doc_id_or_url]
    results = []
    for ident in ids:
        d = _find_doc(ident)
        if d is None:
            continue
        results.append({
            "id": d.url,
            "url": d.url,
            "title": d.title,
            "publishedDate": _published_date_iso(d),
            "text": d.body,
        })
    out: dict = {
        "requestId": _req_id(*ids),
        "results": results,
        "costDollars": {"total": round(0.001 * len(results), 4)},
    }
    if not results:
        out["error"] = f"no content found for id(s): {ids}"
    return out


def _get_current_time() -> str:
    return STATE["now_iso"]


# --------------------------------------------------------------------------- #
# Scoring lives in grader.py (pure function over the submit_answer payload).
# The template records the agent's submit_answer call in STATE["submission"]
# and calls grader.grade(bundle, submission) — no free-text parsing here.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FastMCP tool server + HUD environment
# --------------------------------------------------------------------------- #

server = FastMCP(name="tempora-tools")
env = Environment(name="tempora", version="0.0.1")

_server_task: Optional[asyncio.Task] = None
_port: int = 0


@server.tool
def search(query: str, numResults: int = 10) -> dict:
    """Search the web for pages relevant to the query.

    Returns an Exa-style search response: ``{requestId, results, costDollars}``
    with up to ``numResults`` hits (Exa's default is 10). Each result has ``id``
    (= url), ``url``, ``title``, ``publishedDate`` (ISO or null), ``author``,
    ``score`` (0-1 similarity), ``summary``, and ``highlights``. Results are
    metadata only — call ``fetch`` with the result ``id``(s) to read full text.
    """
    return _search(query, limit=numResults)


@server.tool
def fetch(ids: list[str]) -> dict:
    """Fetch the full text of one or more pages by their ``id`` (the url returned
    by search). Pass a list of ids (Exa's /contents endpoint batches multiple ids
    in one call); a single id is also accepted.

    Returns an Exa-style contents response: ``{requestId, results, costDollars}``
    where each ``results[].text`` is a page body.
    """
    return _fetch(ids)


@server.tool
def get_current_time() -> str:
    """Return the current date/time (ISO 8601). The agent is not told the time
    otherwise — call this to know what 'now' is before judging recency."""
    return _get_current_time()


@server.tool
def submit_answer(items: list[str], supporting_doc_ids: list[str] | None = None) -> dict:
    """Submit your final answer. Call this exactly once, when you are done
    researching — this is the only way your answer is graded; free text in your
    final message is not graded.

    Args:
        items: the names of the items that are currently available / live / open
            right now (the deals, events, or programs you determined are actually
            current as of today's date — excluding anything expired, ended,
            not-yet-open, or undated/unconfirmable).
        supporting_doc_ids: the url(s) (the ``id`` returned by search/fetch) of the
            pages you relied on. An item only counts if you cite a page that
            actually contains it — so cite the source link for each item you list.
    """
    STATE["submission"] = {
        "items": list(items or []),
        "supporting_doc_ids": list(supporting_doc_ids or []),
    }
    return {"status": "recorded", "items": STATE["submission"]["items"],
            "supporting_doc_ids": STATE["submission"]["supporting_doc_ids"]}


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@env.initialize
async def _up() -> None:
    global _server_task, _port
    if _server_task is None:
        _port = _free_port()
        _server_task = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_port)
        )
        # give the server a moment to bind
        await asyncio.sleep(1.0)
    env.add_capability(Capability.mcp(name="tools", url=f"http://127.0.0.1:{_port}/mcp"))


@env.shutdown
async def _down() -> None:
    global _server_task
    if _server_task is not None:
        _server_task.cancel()
        _server_task = None


# --------------------------------------------------------------------------- #
# Task templates — one per query kind. Calling a template mints a concrete Task.
# --------------------------------------------------------------------------- #

DEFAULT_QUERIES = {
    "deal": "deals on AI model subscriptions right now",
    "event": "AI research conferences I can still register for right now",
    "status": "coding bootcamps currently accepting applications",
    "summary": "what's currently true about GPU cloud rental deals",
}


def _make_template(query_kind: str):
    @env.template(id=query_kind, description=f"Tempora {query_kind} temporality task")
    async def _task(
        query: str = DEFAULT_QUERIES[query_kind],
        seed: int = 0,
        difficulty: float = 0.3,
        use_llm: bool = False,
    ):
        bundle = gen.build_bundle(query, seed, difficulty, use_llm=use_llm)
        _load_bundle(bundle)
        _ = yield bundle.prompt  # final message text is not graded
        submission = STATE.get("submission")
        result = grade(bundle, submission)
        yield EvaluationResult(reward=result["reward"], info=result)
    return _task


deal_query = _make_template("deal")
event_query = _make_template("event")
status_query = _make_template("status")
summary_query = _make_template("summary")


# --------------------------------------------------------------------------- #
# Local smoke (no HUD / key needed): exercise the tools + scorer directly.
# --------------------------------------------------------------------------- #

def _local_smoke() -> None:
    b = gen.build_bundle(DEFAULT_QUERIES["deal"], seed=1, difficulty=0.4,
                         force={"aggregator": True, "social": True, "thread_correction": True})
    _load_bundle(b)
    print(f"now (T) = {_get_current_time()}")
    print(f"gold = {[(i, b.item_by_id(i).name) for i in b.gold_ids]}")
    print(f"lure = {[(i, b.item_by_id(i).name, b.item_by_id(i).state.value) for i in b.lure_ids]}")
    print()
    print("search('AI model subscriptions deals') ->")
    res = _search("AI model subscriptions deals")
    print(f"  requestId={res['requestId']}  results={len(res['results'])}  cost=${res['costDollars']['total']}")
    for r in res["results"]:
        pub = r["publishedDate"] or "(null)"
        print(f"  score={r['score']:.2f} pub={pub:26s} {r['title']}")
        print(f"     id={r['id']}  author={r['author']}")
        print(f"     summary: {r['summary'][:80]}")
    print()
    top = res["results"][0]
    print(f"fetch('{top['id'][:40]}...') ->")
    fr = _fetch(top["id"])
    print(f"  requestId={fr['requestId']}  results={len(fr['results'])}")
    body = fr["results"][0]["text"]
    print(f"  publishedDate={fr['results'][0]['publishedDate']}")
    print(f"  text: {body[:160]}...")
    print()
    # batch fetch (Exa /contents shape): pass a list of ids
    all_ids = [r["id"] for r in res["results"]]
    bf = _fetch(all_ids)
    print(f"fetch(<list of {len(all_ids)} ids>) -> results={len(bf['results'])}  cost=${bf['costDollars']['total']}")
    print()
    # --- grader checks (structured submit_answer, pure-function grader) ---
    from grader import grade as _grade

    # map item_id -> a doc url that actually contains it (for grounding)
    def _cite_for(item_id: str) -> str:
        for d in b.docs:
            if any(di.item_id == item_id for di in d.items):
                return d.url
        return b.docs[0].url

    gold = [b.item_by_id(i) for i in b.gold_ids if b.item_by_id(i)]
    lure = [b.item_by_id(i) for i in b.lure_ids if b.item_by_id(i)]
    # perfect: every gold item, each cited to a page that contains it
    perfect = {"items": [it.name for it in gold],
               "supporting_doc_ids": [_cite_for(it.id) for it in gold]}
    print("grade (perfect, cited):", _grade(b, perfect))
    # lure-as-current: assert a lure, cited to a page that contains it -> FP
    bad = {"items": [lure[0].name], "supporting_doc_ids": [_cite_for(lure[0].id)]}
    print("grade (lure-as-current):", _grade(b, bad))
    # bail-to-memorized: correct gold brand names but NO bundle citation
    bail = {"items": [it.name for it in gold], "supporting_doc_ids": []}
    print("grade (bail, no citation):", _grade(b, bail))
    # never called submit_answer
    print("grade (no submission):", _grade(b, None))


if __name__ == "__main__":
    _local_smoke()