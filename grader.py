"""Standalone, programmatic grader for the Tempora temporality environment.

No LLM judge, no free-text parsing. The agent's final answer is a structured
``submit_answer`` payload::

    {"items": [<name>, ...], "supporting_doc_ids": [<url>, ...]}

— the items it asserts are currently available at the reference time T, plus the
corpus doc URLs it relies on. ``grade`` scores that against the bundle's
code-computed ground truth (item state is derived from dates, never an LLM):

  recall     = grounded gold items found / |gold|
  precision  = grounded gold / (grounded gold + grounded lures)
  citation   = cited URLs that are real bundle docs / cited
  format_valid = a well-formed submit_answer was actually called

An item is **grounded** only if the agent cited at least one bundle doc that
actually contains that item — so reciting memorized brand names without citing
the fetched pages scores ~0 (the bail-to-memorized failure mode). This is the
model-independent analogue of the env's old citation-verified free-form scorer:
no prose layout assumptions, so rewards don't depend on the base model's
citation habits.
"""
from __future__ import annotations

import re
from typing import Any

import generator as gen


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def _matches(entity: str, text: str) -> bool:
    """Whole-word match of `entity` in `text` (either direction), so "ChatGPT
    Plus" matches "ChatGPT Plus subscription" but "Pi" doesn't match "picks"."""
    e = _norm(entity)
    t = _norm(text)
    if not e or not t:
        return False
    pat_e = re.compile(r"(?<![A-Za-z0-9])" + re.escape(e) + r"(?![A-Za-z0-9])")
    pat_t = re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])")
    return bool(pat_e.search(t) or pat_t.search(e))


def _doc_item_names(bundle: gen.DocBundle) -> dict[str, set[str]]:
    """url -> set of lowercased item-entity names that appear in that doc."""
    out: dict[str, set[str]] = {}
    for d in bundle.docs:
        names: set[str] = set()
        for di in d.items:
            it = bundle.item_by_id(di.item_id)
            if it is not None:
                names.add(_norm(it.entity))
        out[d.url] = names
    return out


def grade(bundle: gen.DocBundle, submission: dict[str, Any] | None) -> dict[str, Any]:
    """Score a submit_answer payload against `bundle`.

    Returns {"reward", "recall", "precision", "citation_precision",
    "citation_recall", "format_valid"}. If `submission` is missing or malformed
    (the agent never called submit_answer, or it has no "items" key), reward is
    a flat 0.0 and the rest of the calculation is skipped.
    """
    if not submission or "items" not in submission:
        return {
            "reward": 0.0, "recall": 0.0, "precision": 0.0,
            "citation_precision": 0.0, "citation_recall": 0.0,
            "format_valid": False,
        }

    submitted = [_norm(n) for n in submission.get("items") or [] if _norm(n)]
    cited = [u for u in (submission.get("supporting_doc_ids") or []) if u]

    gold = [bundle.item_by_id(i) for i in bundle.gold_ids]
    lures = [bundle.item_by_id(i) for i in bundle.lure_ids]
    gold_names = {_norm(it.entity) for it in gold if it}
    lure_names = {_norm(it.entity) for it in lures if it}

    doc_names = _doc_item_names(bundle)
    cited_doc_names: list[set[str]] = [
        doc_names[u] for u in cited if u in doc_names
    ]  # only real bundle docs the agent cited

    def _grounded(name: str) -> bool:
        # cited: at least one cited bundle doc contains an item matching `name`.
        return any(
            any(_matches(item_name, name) for item_name in names)
            for names in cited_doc_names
        )

    tp = 0  # grounded gold
    fp = 0  # grounded lure
    for n in submitted:
        if not _grounded(n):
            continue  # ungrounded assertion (e.g. memorized, no citation) -> not credited
        if any(_matches(g, n) for g in gold_names):
            tp += 1
        elif any(_matches(l, n) for l in lure_names):
            fp += 1

    recall = tp / len(gold) if gold else 1.0
    denom = tp + fp
    if denom:
        precision = tp / denom
    else:
        precision = 1.0 if not gold else 0.0  # claimed nothing, nothing grounded

    # citation quality (informational; grounding above already enforces citation)
    valid_cited = sum(1 for u in cited if u in doc_names)
    citation_precision = valid_cited / len(cited) if cited else 1.0
    # recall over gold-bearing docs among those cited (did it cite the right pages?)
    gold_doc_urls = {d.url for d in bundle.docs
                     for di in d.items
                     if bundle.item_by_id(di.item_id) and bundle.item_by_id(di.item_id).id in bundle.gold_ids}
    cited_set = set(cited)
    citation_recall = (len(cited_set & gold_doc_urls) / len(gold_doc_urls)
                       if gold_doc_urls else 1.0)

    reward = 0.5 * recall + 0.5 * precision
    return {
        "reward": reward, "recall": recall, "precision": precision,
        "citation_precision": citation_precision, "citation_recall": citation_recall,
        "format_valid": True,
    }