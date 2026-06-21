"""Tempora document generator.

Produces a ``DocBundle`` per ``(query, seed, difficulty)``: a fixed set of
temporally-mixed synthetic documents with *deterministic ground truth* about
which items are active / expired / future / undated_ambiguous at the reference
time ``T``.

Design rules (see plan):
  * The bundle is generated once per task, before the agent acts. ``search``
    ranks this fixed corpus; it never generates new docs.
  * The temporal skeleton (dates, offsets, conflicts, item states) is produced
    by a seeded ``random.Random`` so ground truth is known exactly.
  * An optional LLM ``proseify`` path can dress the skeleton in richer prose,
    but the skeleton + ground truth are always code-owned. Default mode is
    ``--no-prose``: deterministic template prose, no API key required.
  * Item state is source-independent: an item's state at T follows from its own
    dates, regardless of which doc it appears in. A gold item can live inside a
    stale aggregator; a lure can live in a primary source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

# A fixed anchor so generation is reproducible across runs/machines. The
# reference time T for a bundle is this anchor plus a seed-derived offset, so T
# varies across bundles (calendar-context variety) but is fully deterministic.
BASE_NOW = datetime(2026, 6, 20, 12, 0, 0)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class ItemState(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    FUTURE = "future"
    UNDATED_AMBIGUOUS = "undated_ambiguous"


class ItemKind(str, Enum):
    DEAL = "deal"        # has valid_from + expires_at
    EVENT = "event"      # has start + end
    STATUS = "status"    # has window_open + window_close ("accepting applications")


class LabelForm(str, Enum):
    EXPLICIT = "explicit"   # "March 14, 2026"
    RELATIVE = "relative"   # "3 days ago", "next Tuesday"
    VAGUE = "vague"         # "recently", "soon", "this season"
    MISSING = "missing"     # date omitted entirely


# Source types. aggregator_list and social_post carry their own temporal traps.
SOURCE_TYPES = ("news", "forum", "marketing", "pdf_receipt", "aggregator_list", "social_post")
SOCIAL_SUBTYPES = ("tweet", "reddit", "hackernews", "linkedin")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class Item:
    """Canonical item with true dates and computed state at T. Source-independent."""
    id: str
    kind: ItemKind
    name: str
    entity: str
    # Temporal extents (None => unknown/undated for that bound).
    valid_from: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    window_open: Optional[datetime] = None
    window_close: Optional[datetime] = None
    state: ItemState = ItemState.UNDATED_AMBIGUOUS
    currency_phrase: str = ""     # e.g. "right now", "still live" — prose framing
    # Realistic listing details (deterministic from rng). A real listicle names a
    # price, a discount, a promo code, billing terms, and a one-sentence blurb per
    # pick — without these the pages read as thin/auto-generated and the agent
    # flags them. Blurbs are entity- and competitor-name-free and avoid temporal
    # cue words so they don't perturb the scorer.
    price: str = ""          # e.g. "$20/mo" (deal), "$1,200" (event reg), "free"
    list_price: str = ""     # pre-discount price, e.g. "$33/mo"
    discount: str = ""       # e.g. "40% off", "3 months free"
    promo_code: str = ""     # e.g. "AI40"
    terms: str = ""          # e.g. "new subscribers, annual billing"
    blurb: str = ""          # one descriptive sentence
    # Events: real-sounding provenance so the page doesn't read as a bare
    # auto-generated listing (the agent flagged "no organizer, no registration
    # link" as a fabrication tell).
    organizer: str = ""      # e.g. "NeurIPS Foundation", "MLCommons"
    reg_url: str = ""        # a registration/program URL on the bundle domain set

    def date_fields(self) -> dict:
        return {
            "valid_from": self.valid_from, "expires_at": self.expires_at,
            "start": self.start, "end": self.end,
            "window_open": self.window_open, "window_close": self.window_close,
        }


@dataclass
class DocItem:
    """An item as it appears in one particular doc: which dates are shown, in
    what label form, and any conflicting override text."""
    item_id: str
    show_fields: tuple[str, ...]          # which of the item's dates this doc displays
    label_forms: dict[str, LabelForm]     # field -> label form
    override_labels: dict[str, str] = field(default_factory=dict)  # field -> wrong text (conflict)


@dataclass
class Reply:
    user: str
    text: str
    is_correction: bool = False


@dataclass
class Doc:
    doc_id: str
    url: str
    title: str
    source_type: str
    source_subtype: str = ""
    published_at: Optional[datetime] = None
    published_label_form: LabelForm = LabelForm.MISSING
    published_label: str = ""            # rendered string the agent sees
    items: list[DocItem] = field(default_factory=list)
    body: str = ""                       # rendered prose (template or LLM)
    search_rank_hint: float = 0.5        # higher = ranks earlier in search()
    replies: list[Reply] = field(default_factory=list)
    engagement: dict = field(default_factory=dict)   # upvotes/likes (social decoys)
    edited: bool = False
    snippet: str = ""


@dataclass
class DocBundle:
    now: datetime
    prompt: str
    query: str
    query_kind: str
    docs: list[Doc]
    items: list[Item]
    gold_ids: list[str]
    lure_ids: list[str]

    # ---- convenience for the environment / scorer ----
    def doc_by_id(self, doc_id: str) -> Optional[Doc]:
        for d in self.docs:
            if d.doc_id == doc_id:
                return d
        return None

    def item_by_id(self, item_id: str) -> Optional[Item]:
        for it in self.items:
            if it.id == item_id:
                return it
        return None

    def to_json(self) -> str:
        return json.dumps(_bundle_to_dict(self), indent=2, default=str)


# --------------------------------------------------------------------------- #
# Entity / topic pools (value variety)
# --------------------------------------------------------------------------- #

DEAL_TOPICS = [
    "AI model subscriptions", "GPU cloud rentals", "noise-cancelling headphones",
    "mechanical keyboards", "flights to Tokyo", "SaaS lifetime deals",
    "standing desks", "espresso machines", "trail running shoes", "VR headsets",
]
# Generic real-brand fallbacks (used only when a topic has no entry in
# TOPIC_ENTITIES below — every topic the 14 queries hit is covered there, so
# these are a safety net for ad-hoc queries). Real names only: invented brands
# ("ApexAI", "Quanta", ...) made agents dismiss the corpus as fabricated.
DEAL_ENTITIES = [
    "OpenAI", "Google", "Microsoft", "Anthropic", "Meta",
    "Amazon", "Apple", "NVIDIA", "Samsung", "Adobe",
    "Slack", "Zoom", "Notion", "Figma", "Stripe",
]
EVENT_TOPICS = [
    "AI research conference", "startup bootcamp", "developer meetup", "pitch night",
    "hardware hackathon", "design workshop", "robotics demo day", "investor summit",
    "LLM fine-tuning webinar", "founder dinner",
]
EVENT_ENTITIES = [
    "Web Summit", "SXSW", "CES", "Dreamforce", "AWS re:Invent",
    "Google I/O", "Microsoft Build", "Apple WWDC", "Disrupt", "Collision",
    "Slush", "TwitchCon", "E3", "ReactConf", "KubeCon",
]
STATUS_TOPICS = [
    "job postings", "grant applications", "beta program access", "coding bootcamp cohort",
    "accelerator applications", "restaurant opening", "fellowship applications",
    "scholarship applications", "call for papers", "early-access waitlist",
]
STATUS_ENTITIES = [
    "Google", "Microsoft", "Amazon", "OpenAI", "Meta",
    "Stripe", "Notion", "Figma", "GitLab", "Atlassian",
    "Salesforce", "HubSpot", "Zoom", "Slack", "Dropbox",
]

# Real, topic-specific brand/product/org names keyed by the infer_topic() topic
# string. Each pool has 15 distinct real names (>= max n_items=15) so sampled
# item names are always real and unique. This is the fix for the "fabricated
# brand names" bail: the corpus now names ChatGPT Plus, Sony WH-1000XM5, NeurIPS,
# Y Combinator, Meta Quest 3, etc. — products/orgs the agent recognizes, so it
# engages with the pages instead of dismissing them as synthetic.
TOPIC_ENTITIES: dict[str, list[str]] = {
    # ---- deals ----
    "AI model subscriptions": [
        "ChatGPT Plus", "Claude Pro", "Gemini Advanced", "Copilot Pro", "Perplexity Pro",
        "Grok", "Midjourney", "Cursor Pro", "GitHub Copilot", "Notion AI",
        "Runway", "Suno Pro", "Pika", "DeepSeek", "Pi",
    ],
    "GPU cloud rentals": [
        "RunPod", "Lambda Labs", "Vast.ai", "CoreWeave", "Paperspace",
        "Modal", "Replicate", "Baseten", "TensorDock", "Together AI",
        "Jarvis Labs", "Featherless", "Lightning AI", "Anyscale", "Crusoe",
    ],
    "noise-cancelling headphones": [
        "Sony WH-1000XM5", "Bose QuietComfort Ultra", "Apple AirPods Max", "Sennheiser Momentum 4", "Sony WF-1000XM5",
        "Bose QC Earbuds", "JBL Tour One M2", "AirPods Pro 2", "Bowers & Wilkins Px7 S2", "Technics EAH-A800",
        "Beats Studio Pro", "Shure Aonic 50", "Soundcore Space One", "AKG N700M2", "Mark Levinson No. 5909",
    ],
    "mechanical keyboards": [
        "Keychron Q1", "HHKB Hybrid", "Ducky One 3", "Razer Huntsman V3", "Corsair K70",
        "Wooting 60HE", "NuPhy Halo75", "Mode Envoy", "Keydous NJ80", "Leopold FC660C",
        "Realforce R3", "Glorious GMMK Pro", "Drop CTRL", "Akko 5075", "Luminkey75",
    ],
    "flights to Tokyo": [
        "ANA", "JAL", "United", "Delta", "American",
        "Cathay Pacific", "Singapore Airlines", "Korean Air", "Air Canada", "Zipair",
        "Hawaiian Airlines", "British Airways", "Lufthansa", "Etihad", "Qatar Airways",
    ],
    "SaaS lifetime deals": [
        "TidyCal", "SendFox", "KingSumo", "Paperform", "Softr",
        "Tally", "Senja", "Bannerbear", "Nifty", "Stackby",
        "Formaloo", "Elfsight", "Beamer", "Better Stack", "Tagbox",
    ],
    "standing desks": [
        "Vari", "Uplift V2", "Fully Jarvis", "Branch Ergonomic Desk", "FlexiSpot E7",
        "Secretlab Magnus", "IKEA Bekant", "Autonomous SmartDesk", "Stand Desk Pro", "Ergonofis Shift",
        "Desky Dual", "Branch Standing Desk", "Loctek E7", "Workstream", "MOES",
    ],
    "espresso machines": [
        "Breville Barista Express", "DeLonghi La Specialista", "Gaggia Classic Pro", "Rancilio Silvia", "ECM Mechanika",
        "Lelit Anna", "Profitec Pro 600", "La Marzocco Micra", "Bambino Plus", "Rocket Appartamento",
        "Sage Oracle", "Nuova Simonelli Oscar", "Eureka Mignon", "Nivona", "Jura E8",
    ],
    "trail running shoes": [
        "Hoka Speedgoat 5", "Salomon Sense Ride", "Brooks Cascadia", "Altra Lone Peak", "Saucony Peregrine",
        "La Sportiva Bushido", "Inov-8 Terraultra", "Nike Pegasus Trail", "Asics Gel-Venture", "Merrell MTL Long Sky",
        "Nnormal Tomir", "Vibram FiveFingers", "Scarpa Spin", "On Cloudventure", "Adidas Terrex",
    ],
    "VR headsets": [
        "Meta Quest 3", "Apple Vision Pro", "PSVR2", "Meta Quest Pro", "Bigscreen Beyond",
        "Pimax Crystal", "Valve Index", "HTC Vive XR Elite", "Meta Quest 2", "Pico 4",
        "Varjo XR-4", "Somnium VR1", "Lynx R-1", "HTC Vive Pro 2", "DPVR E4",
    ],
    # ---- events ----
    "AI research conference": [
        "NeurIPS", "ICML", "ICLR", "CVPR", "ACL",
        "EMNLP", "AAAI", "IJCAI", "KDD", "MLSys",
        "COLM", "DeepLearningIndaba", "RAAIS", "AI@Scale", "TinyML Summit",
    ],
    "startup bootcamp": [
        "YC Startup School", "Techstars Startup Weekend", "Founder Institute", "RebelBio", "SU Labs",
        "Launchpad LA", "500 Bootcamp", "Outlier Campus", "Antler Bootcamp", "On Deck Founder",
        "Beta Boom Bootcamp", "AlchemistX", "Surge Bootcamp", "Founder University", "Startup School",
    ],
    "developer meetup": [
        "ReactConf", "JSConf", "RustConf", "KubeCon", "DockerCon",
        "Devoxx", "Velocity", "Strange Loop", "NodeConf", "GoCon",
        "Kafka Summit", "EmberConf", "ElixirConf", "DotPy", "PyCon",
    ],
    "pitch night": [
        "YC Demo Day", "Techstars Demo Day", "500 Demo Day", "StartX Demo Day", "AngelPad Demo Day",
        "Capital Factory Demo Day", "MassChallenge Demo Day", "Surge Demo Day", "AlchemistX Demo Day", "Antler Demo Day",
        "Founders Fund Demo", "Boomtown Demo", "Catapult Demo", "Launch Demo", "C100 Demo",
    ],
    "hardware hackathon": [
        "TreeHacks", "PennApps", "MHacks", "HackMIT", "HackTheNorth",
        "BigRedHacks", "HackTX", "BoilerMake", "HackBeanpot", "SteelHacks",
        "HackIllinois", "HackUCI", "HackDuke", "HackBerkeley", "HackBean",
    ],
    "design workshop": [
        "Awwwards Conference", "Adobe MAX", "Config by Figma", "Brand New Conference", "DesignOps Summit",
        "AIGA Conference", "ConveyUX", "UXDX", "SmashingConf", "Design+Research",
        "AWF", "IxDA Interaction", "Layered", "Justified", "Dribbble Meetup",
    ],
    "robotics demo day": [
        "Skydio Demo", "Boston Dynamics Demo", "Figure AI Demo", "1X Demo", "Agility Robotics Demo",
        "Covariant Demo", "Berkshire Grey Demo", "Nuro Demo", "Zipline Demo", "FarmWise Demo",
        "Bossa Nova Demo", "Fetch Robotics Demo", "Locus Demo", "Symbotic Demo", "AMP Robotics Demo",
    ],
    "investor summit": [
        "Web Summit", "Slush", "Collision", "Disrupt", "SOSV Summit",
        "YC Investor Day", "Angel Summit", "Midas List Summit", "Techstars Summit", "Singularity Summit",
        "OurCrowd Summit", "F50 Summit", "Global Capital Summit", "VentureSummit", "Capgemini Summit",
    ],
    "LLM fine-tuning webinar": [
        "Hugging Face webinar", "Weights & Biases webinar", "DeepLearning.AI webinar", "Modal webinar", "Together AI webinar",
        "LangChain webinar", "LlamaIndex webinar", "Cohere webinar", "Anyscale webinar", "Predibase webinar",
        "OpenAI DevDay", "Mistral webinar", "Determined AI webinar", "Ray webinar", "Lightning AI webinar",
    ],
    "founder dinner": [
        "YC Founder Dinner", "Techstars Founder Dinner", "Startup Grind", "Founders Network Dinner", "Indiegogo Founder Night",
        "EO Dinner", "Vistage Dinner", "Tiger 21 Dinner", "YC Alumni Dinner", "500 Founder Dinner",
        "On Deck Dinner", "Antler Founder Dinner", "Alchemist Dinner", "C100 Dinner", "Founders Dinner NYC",
    ],
    # ---- status ----
    "job postings": [
        "Google", "Microsoft", "Apple", "Amazon", "Meta",
        "NVIDIA", "OpenAI", "Anthropic", "Stripe", "Notion",
        "Figma", "GitLab", "Databricks", "Snowflake", "Coinbase",
    ],
    "grant applications": [
        "NSF SBIR", "NIH R01", "DOE SBIR", "DARPA", "Sloan Foundation",
        "Knight Foundation", "Mozilla MOSS", "NVIDIA Inception", "AWS EdStart", "Google for Startups",
        "Microsoft AI for Good", "OpenAI Researcher Access", "Anthropic Academic Grant", "Cohere Grant", "Hugging Face Grant",
    ],
    "beta program access": [
        "OpenAI Beta", "Google Labs", "Microsoft Preview", "Anthropic Preview", "Meta Beta",
        "Notion Early Access", "Figma Preview", "Linear Beta", "Vercel Beta", "Supabase Beta",
        "Replit Bounties", "Raycast Beta", "Arc Early Access", "Warp Preview", "Cursor Preview",
    ],
    "coding bootcamp cohort": [
        "General Assembly", "Flatiron School", "Hack Reactor", "Fullstack Academy", "Springboard",
        "Coding Dojo", "BrainStation", "Tech Elevator", "Rithm School", "Launch School",
        "Codesmith", "BloomTech", "App Academy", "Epicodus", "Thinkful",
    ],
    "accelerator applications": [
        "Y Combinator", "Techstars", "500 Global", "AngelPad", "AlchemistX",
        "Surge", "a16z CSX", "Antler", "Founder University", "Startup School",
        "On Deck", "Beta Boom", "Capital Factory", "gBETA", "Catalyst",
    ],
    "restaurant opening": [
        "McDonald's", "Chipotle", "Sweetgreen", "Shake Shack", "Cava",
        "First Watch", "Cracker Barrel", "Tender Greens", "True Food Kitchen", "Veggie Grill",
        "&pizza", "Blaze Pizza", "Dave's Hot Chicken", "Wingstop", "Portillo's",
    ],
    "fellowship applications": [
        "Echoing Green", "Knight-Hennessy", "Schwarzman Scholars", "Rhodes", "Ford Foundation Fellowship",
        "Obama Foundation Fellowship", "New America Fellowship", "Aspen Institute Fellowship", "Clarendon Fund", "Cambridge Trust",
        "Fulbright", "DAAD", "Endeavor Fellowship", "Mozilla Fellowship", "AAAS Fellowship",
    ],
    "scholarship applications": [
        "Rhodes", "Marshall", "Truman", "Gates Millennium", "Fulbright",
        "Chevening", "Schwarzman", "Knight-Hennessy", "Cameron Impact", "Coca-Cola Scholars",
        "Davidson Fellows", "Regeneron STS", "Burger King Scholars", "AXA Achievement", "Horatio Alger",
    ],
    "call for papers": [
        "NeurIPS", "ICML", "ICLR", "CVPR", "ACL",
        "EMNLP", "AAAI", "IJCAI", "KDD", "WWW",
        "SIGIR", "SIGCHI", "OSDI", "SOSP", "PLDI",
    ],
    "early-access waitlist": [
        "OpenAI waitlist", "Google Labs waitlist", "Apple waitlist", "Meta waitlist", "Anthropic waitlist",
        "Notion AI waitlist", "Rabbit r1 waitlist", "Humane AI Pin waitlist", "xAI Grok waitlist", "Worldcoin waitlist",
        "Pi waitlist", "Inflection waitlist", "Character.AI waitlist", "Perplexity waitlist", "Sora waitlist",
    ],
}

# Real conference months (the memorized schedule a capable agent knows and uses
# to spot fabricated dates — NeurIPS is December, not September). Pinning these
# keeps the page consistent with the agent's world model. State (active/expired/
# future) is still controlled by the registration window relative to T, so any
# conference can fill any state slot while its event date stays in the real month.
_CONFERENCE_MONTHS: dict[str, int] = {
    "NeurIPS": 12, "ICML": 7, "ICLR": 5, "CVPR": 6, "ACL": 7,
    "EMNLP": 11, "AAAI": 2, "IJCAI": 8, "KDD": 8, "MLSys": 6,
    "COLM": 6, "DeepLearningIndaba": 8, "RAAIS": 7, "AI@Scale": 7, "TinyML Summit": 5,
}
_CONFERENCE_ORGANIZERS: dict[str, str] = {
    "NeurIPS": "NeurIPS Foundation", "ICML": "ICML", "ICLR": "ICLR",
    "CVPR": "CVPR Organizing Committee", "ACL": "Association for Computational Linguistics",
    "EMNLP": "ACL Special Interest Group on Natural Language Processing",
    "AAAI": "Association for the Advancement of Artificial Intelligence",
    "IJCAI": "IJCAI", "KDD": "ACM SIGKDD", "MLSys": "MLSys",
    "COLM": "COLM", "DeepLearningIndaba": "Deep Learning Indaba",
    "RAAIS": "Applied AI Institute", "AI@Scale": "AI@Scale Community", "TinyML Summit": "TinyML Foundation",
}
_CONFERENCE_DURATIONS: dict[str, int] = {
    "NeurIPS": 7, "ICML": 7, "ICLR": 5, "CVPR": 6, "ACL": 6,
    "EMNLP": 6, "AAAI": 7, "IJCAI": 6, "KDD": 6, "MLSys": 5,
    "COLM": 5, "DeepLearningIndaba": 6, "RAAIS": 2, "AI@Scale": 3, "TinyML Summit": 3,
}

CURRENCY_PHRASES_CURRENT = ["right now", "currently", "still live", "just dropped", "available now", "going on now"]
CURRENCY_PHRASES_FUTURE = ["coming soon", "launching soon", "drops soon", "opening soon"]
CURRENCY_PHRASES_PAST = ["was live", "recently wrapped", "just ended", "previously available"]

VAGUE_PAST = ["recently", "a while back", "this season", "earlier this year", "not long ago", "a few months back"]
VAGUE_FUTURE = ["soon", "in the near future", "this coming season", "any day now", "shortly"]
VAGUE_AMBIG = ["lately", "these days", "around now", "currently-ish"]


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #

def _seed_hash(query: str, seed: int) -> int:
    h = hashlib.sha256(f"{query}\x1f{seed}".encode()).hexdigest()
    return int(h, 16)


def explicit_phrase(d: datetime) -> str:
    return d.strftime("%B %-d, %Y") if hasattr(d, "strftime") else str(d)


def relative_phrase(d: datetime, T: datetime) -> str:
    """Render a datetime as a human relative phrase anchored at T."""
    delta_days = (d.date() - T.date()).days
    if delta_days == 0:
        return "today"
    if delta_days == -1:
        return "yesterday"
    if delta_days == 1:
        return "tomorrow"
    if -7 < delta_days < 0:
        return f"{-delta_days} days ago"
    if 0 < delta_days < 7:
        return f"in {delta_days} days"
    if -28 < delta_days < 0:
        w = max(1, -delta_days // 7)
        return f"about {w} week{'s' if w > 1 else ''} ago"
    if 0 < delta_days < 28:
        w = max(1, delta_days // 7)
        return f"in about {w} week{'s' if w > 1 else ''}"
    if delta_days < 0:
        return f"back in {d.strftime('%B %Y')}"
    return f"sometime in {d.strftime('%B %Y')}"


def vague_phrase(d: Optional[datetime], T: datetime, rng: random.Random) -> str:
    if d is None:
        return rng.choice(VAGUE_AMBIG)
    if d < T:
        return rng.choice(VAGUE_PAST)
    return rng.choice(VAGUE_FUTURE)


def render_label(d: Optional[datetime], T: datetime, form: LabelForm, rng: random.Random) -> str:
    """Render a date in the given label form. MISSING => empty string."""
    if form == LabelForm.MISSING or d is None:
        return ""
    if form == LabelForm.EXPLICIT:
        return explicit_phrase(d)
    if form == LabelForm.RELATIVE:
        return relative_phrase(d, T)
    return vague_phrase(d, T, rng)


def _pick_label_form(rng: random.Random, difficulty: float, prefer_missing: bool = False) -> LabelForm:
    """Difficulty raises the chance of vague/missing labels (less temporal clarity)."""
    if prefer_missing:
        weights = [0.1, 0.1, 0.2, 0.6]
    else:
        # explicit, relative, vague, missing
        weights = [
            max(0.05, 0.55 - 0.35 * difficulty),
            max(0.05, 0.25),
            0.10 + 0.20 * difficulty,
            0.05 + 0.25 * difficulty,
        ]
    return rng.choices(list(LabelForm), weights=weights, k=1)[0]


# --------------------------------------------------------------------------- #
# State computation (the single source of truth)
# --------------------------------------------------------------------------- #

def compute_state(item: Item, T: datetime) -> ItemState:
    if item.kind == ItemKind.DEAL:
        vf, ex = item.valid_from, item.expires_at
        if vf is None and ex is None:
            return ItemState.UNDATED_AMBIGUOUS
        if vf is not None and T < vf:
            return ItemState.FUTURE
        if ex is not None and T > ex:
            return ItemState.EXPIRED
        return ItemState.ACTIVE
    if item.kind == ItemKind.EVENT:
        # State tracks the registration window (active = reg open now). Fall back
        # to event dates only when no reg window is set (e.g. undated items).
        o, c = item.window_open, item.window_close
        if o is not None or c is not None:
            if o is not None and T < o:
                return ItemState.FUTURE
            if c is not None and T > c:
                return ItemState.EXPIRED
            return ItemState.ACTIVE
        s, e = item.start, item.end
        if s is None and e is None:
            return ItemState.UNDATED_AMBIGUOUS
        if s is not None and T < s:
            return ItemState.FUTURE
        if e is not None and T > e:
            return ItemState.EXPIRED
        return ItemState.ACTIVE
    if item.kind == ItemKind.STATUS:
        o, c = item.window_open, item.window_close
        if o is None and c is None:
            return ItemState.UNDATED_AMBIGUOUS
        if o is not None and T < o:
            return ItemState.FUTURE
        if c is not None and T > c:
            return ItemState.EXPIRED
        return ItemState.ACTIVE
    return ItemState.UNDATED_AMBIGUOUS


# --------------------------------------------------------------------------- #
# Item construction with a target state
# --------------------------------------------------------------------------- #

def _conference_start_on_or_after(entity: str, after: datetime,
                                  rng: random.Random) -> datetime:
    """Nearest real-month occurrence of `entity` whose start >= after (deterministic
    given T/entity/order; advances rng by one draw per year scanned)."""
    m = _CONFERENCE_MONTHS[entity]
    year = after.year
    while True:
        day = rng.randint(2, 25)
        start = datetime(year, m, day)
        if start >= after:
            return start
        year += 1


def _conference_start_before(entity: str, before: datetime,
                             rng: random.Random) -> datetime:
    """Most recent real-month occurrence of `entity` whose end < before."""
    m = _CONFERENCE_MONTHS[entity]
    dur = _CONFERENCE_DURATIONS.get(entity, 5)
    year = before.year
    while True:
        day = rng.randint(2, 25)
        start = datetime(year, m, day)
        if start + timedelta(days=dur) < before:
            return start
        year -= 1


def _make_item(rng: random.Random, idx: int, kind: ItemKind, target: ItemState, T: datetime,
               topic: str, entity: str) -> Item:
    if kind == ItemKind.DEAL:
        # The doc title/snippet already carries the topic ("Top N AI model
        # subscription deals right now"); the item line just names the product,
        # like a real listing: "ChatGPT Plus (right now) [expires: ...]".
        name = entity
        it = Item(id=f"item-{idx}", kind=kind, name=name, entity=entity)
        if target == ItemState.ACTIVE:
            it.valid_from = T - timedelta(days=rng.randint(1, 30))
            it.expires_at = T + timedelta(days=rng.randint(1, 60))
            it.currency_phrase = rng.choice(CURRENCY_PHRASES_CURRENT)
        elif target == ItemState.EXPIRED:
            it.valid_from = T - timedelta(days=rng.randint(60, 180))
            it.expires_at = T - timedelta(days=rng.randint(1, 45))
            it.currency_phrase = rng.choice(CURRENCY_PHRASES_PAST)
        elif target == ItemState.FUTURE:
            it.valid_from = T + timedelta(days=rng.randint(5, 60))
            it.expires_at = it.valid_from + timedelta(days=rng.randint(7, 90))
            it.currency_phrase = rng.choice(CURRENCY_PHRASES_FUTURE)
        else:  # UNDATED_AMBIGUOUS
            it.currency_phrase = rng.choice(CURRENCY_PHRASES_CURRENT)  # misleading
        return it

    if kind == ItemKind.EVENT:
        # Bare conference/meetup name ("NeurIPS"); the doc title gives context.
        # State is driven by the REGISTRATION window (active = reg open now, which
        # is what "conferences I can still register for right now" asks). Event
        # dates carry the real month for known conferences so a capable agent's
        # memorized schedule (NeurIPS=December) matches the page instead of
        # contradicting it.
        name = entity
        it = Item(id=f"item-{idx}", kind=kind, name=name, entity=entity)
        is_conf = entity in _CONFERENCE_MONTHS
        dur = _CONFERENCE_DURATIONS.get(entity, 5) if is_conf else 0
        if target == ItemState.ACTIVE:
            # reg open NOW (window straddles T); event happens after reg closes.
            it.window_open = T - timedelta(days=rng.randint(10, 40))
            it.window_close = T + timedelta(days=rng.randint(3, 25))
            if is_conf:
                it.start = _conference_start_on_or_after(
                    entity, it.window_close + timedelta(days=1), rng)
                it.end = it.start + timedelta(days=dur)
            else:
                it.start = it.window_close + timedelta(days=rng.randint(2, 20))
                it.end = it.start + timedelta(days=rng.randint(1, 4))
            it.currency_phrase = rng.choice(CURRENCY_PHRASES_CURRENT)
        elif target == ItemState.EXPIRED:
            # event already happened / reg closed — all dates in the past.
            if is_conf:
                it.start = _conference_start_before(entity, T, rng)
                it.end = it.start + timedelta(days=dur)
            else:
                it.start = T - timedelta(days=rng.randint(20, 120))
                it.end = it.start + timedelta(days=rng.randint(1, 5))
            it.window_close = it.start - timedelta(days=rng.randint(1, 10))
            it.window_open = it.window_close - timedelta(days=rng.randint(60, 100))
            it.currency_phrase = rng.choice(CURRENCY_PHRASES_PAST)
        elif target == ItemState.FUTURE:
            # announced but reg not open yet — all dates in the future.
            it.window_open = T + timedelta(days=rng.randint(10, 40))
            it.window_close = it.window_open + timedelta(days=rng.randint(20, 50))
            if is_conf:
                it.start = _conference_start_on_or_after(
                    entity, it.window_close + timedelta(days=1), rng)
                it.end = it.start + timedelta(days=dur)
            else:
                it.start = it.window_close + timedelta(days=rng.randint(7, 40))
                it.end = it.start + timedelta(days=rng.randint(1, 4))
            it.currency_phrase = rng.choice(CURRENCY_PHRASES_FUTURE)
        else:
            it.currency_phrase = rng.choice(CURRENCY_PHRASES_CURRENT)
        return it

    # STATUS
    # Bare org/program name ("Y Combinator"); the doc title gives context.
    name = entity
    it = Item(id=f"item-{idx}", kind=kind, name=name, entity=entity)
    if target == ItemState.ACTIVE:
        it.window_open = T - timedelta(days=rng.randint(1, 30))
        it.window_close = T + timedelta(days=rng.randint(1, 45))
        it.currency_phrase = rng.choice(CURRENCY_PHRASES_CURRENT)
    elif target == ItemState.EXPIRED:
        it.window_open = T - timedelta(days=rng.randint(60, 150))
        it.window_close = T - timedelta(days=rng.randint(1, 30))
        it.currency_phrase = rng.choice(CURRENCY_PHRASES_PAST)
    elif target == ItemState.FUTURE:
        it.window_open = T + timedelta(days=rng.randint(7, 60))
        it.window_close = it.window_open + timedelta(days=rng.randint(14, 60))
        it.currency_phrase = rng.choice(CURRENCY_PHRASES_FUTURE)
    else:
        it.currency_phrase = rng.choice(CURRENCY_PHRASES_CURRENT)
    return it


# --------------------------------------------------------------------------- #
# Realistic listing details (price / discount / promo / terms / blurb) per
# topic. These are what make a page read like a real listicle instead of a
# thin auto-generated bullet list — the agent flagged "no pricing or discount
# amounts" as the giveaway. Blurbs are entity- and competitor-name-free and
# avoid temporal-cue words so they don't perturb the scorer.
# --------------------------------------------------------------------------- #

_DEAL_PRICES: dict[str, list[tuple[str, str]]] = {
    "AI model subscriptions": [("$20/mo", "$25/mo"), ("$17/mo", "$20/mo"), ("$200/yr", "$300/yr"),
        ("$10/mo", "$20/mo"), ("$30/mo", "$40/mo"), ("$8/mo", "$12/mo"), ("$15/mo", "$20/mo"),
        ("$100/yr", "$192/yr")],
    "GPU cloud rentals": [("$0.40/hr", "$0.80/hr"), ("$0.30/hr", "$0.50/hr"), ("$1.20/hr", "$2.00/hr"),
        ("$0.75/hr", "$1.10/hr"), ("$250/mo", "$400/mo"), ("$0.55/hr", "$0.90/hr")],
    "noise-cancelling headphones": [("$279", "$349"), ("$329", "$399"), ("$499", "$549"),
        ("$249", "$299"), ("$199", "$249"), ("$379", "$429")],
    "mechanical keyboards": [("$149", "$199"), ("$220", "$260"), ("$179", "$210"),
        ("$129", "$159"), ("$310", "$360")],
    "flights to Tokyo": [("$680", "$1,100"), ("$890", "$1,400"), ("$1,150", "$1,800"),
        ("$720", "$1,200"), ("$950", "$1,500")],
    "SaaS lifetime deals": [("$69 one-time", "$240/yr"), ("$129 one-time", "$600/yr"),
        ("$49 one-time", "$180/yr"), ("$199 one-time", "$960/yr")],
    "standing desks": [("$395", "$595"), ("$549", "$699"), ("$299", "$399"),
        ("$725", "$899"), ("$449", "$599")],
    "espresso machines": [("$649", "$799"), ("$899", "$1,099"), ("$1,495", "$1,795"),
        ("$449", "$549"), ("$2,395", "$2,895")],
    "trail running shoes": [("$145", "$170"), ("$110", "$140"), ("$160", "$180"),
        ("$95", "$120"), ("$130", "$150")],
    "VR headsets": [("$499", "$549"), ("$3,499", "$3,999"), ("$549", "$599"),
        ("$999", "$1,099"), ("$299", "$399")],
}
_DEAL_PRICES_GENERIC = [("$29", "$39"), ("$49", "$69"), ("$19", "$25"), ("$79", "$99"), ("$129", "$159")]

_DEAL_DISCOUNTS = ["40% off", "Save 30%", "3 months free", "$50 off", "Half off the first year",
    "Save $10/mo", "25% off annual", "60% off", "Buy one get one free"]
_DEAL_PROMOS = ["AI40", "DEAL10", "SUB50", "HELM50", "EARLY30", "SAVE25", "FALL40", "NEW15", "GROW20"]
_DEAL_TERMS = ["new subscribers", "annual billing", "first year only", "students only",
    "first 1,000 signups", "12-month commitment", "first 3 months", "US customers"]

# Deal details must match the product type, or the page reads as auto-generated
# (e.g. "25% off annual plan" on a pair of headphones is nonsense). Split by
# category: recurring subscriptions, one-time SaaS lifetime deals, physical
# goods, and flights each get their own discount/term vocabulary.
_DEAL_PHYSICAL_TOPICS = {"noise-cancelling headphones", "mechanical keyboards", "standing desks",
    "espresso machines", "trail running shoes", "VR headsets"}
_DEAL_FLIGHTS_TOPICS = {"flights to Tokyo"}
_DEAL_LIFETIME_TOPICS = {"SaaS lifetime deals"}

# Real-ish registration/program domains for known conferences (the agent flagged
# "no verifiable registration links" as a tell). Non-conference events get a
# generic eventbrite/lu.ma-style URL.
_CONFERENCE_REG_DOMAINS: dict[str, str] = {
    "NeurIPS": "neurips.cc", "ICML": "icml.cc", "ICLR": "iclr.cc",
    "CVPR": "cvpr.thecvf.com", "ACL": "aclweb.org", "EMNLP": "aclweb.org",
    "AAAI": "aaai.org", "IJCAI": "ijcai.org", "KDD": "kdd.org",
    "MLSys": "mlsys.org", "COLM": "colmweb.org", "DeepLearningIndaba": "deeplearningindaba.com",
    "RAAIS": "appliedai-institute.org", "AI@Scale": "aiscale.io", "TinyML Summit": "tinyml.org",
}
_EVENT_REG_DOMAINS = ["eventbrite.com", "lu.ma", "conference.inc", "events.humanitix.com"]
_EVENT_ORGANIZERS: dict[str, list[str]] = {
    "developer meetup": ["Local dev community", "City Developers", "Tech Meetup Group"],
    "LLM fine-tuning webinar": ["MLCommons", "The AI Native Dev", "MLOps Community"],
    "hardware hackathon": ["Major League Hacking", "Hack Club", "Hardware.dev"],
    "pitch night": ["Founders Network", "Startup Grind", "AngelHack"],
    "design workshop": ["AIGA", "DesignLab", "IxDA"],
    "robotics demo day": ["Robotics Hub", "MassRobotics", " Automate.org"],
    "investor summit": ["Summit Partners", "Tech Coast Conference", "GCV Summit"],
    "startup bootcamp": ["YC Startup School", "Techstars", "Founder Institute"],
    "founder dinner": ["On Deck", "Founders Dinners", "DinnerLab"],
}
_EVENT_ORGANIZERS_GENERIC = ["The organizing committee"]
_DEAL_DISCOUNTS_SUB = ["3 months free", "Save $10/mo", "25% off annual", "40% off", "Save 30%",
    "Half off the first year", "60% off", "$50 off"]
_DEAL_TERMS_SUB = ["new subscribers", "annual billing", "first year only", "students only",
    "first 3 months", "12-month commitment", "US customers"]
_DEAL_DISCOUNTS_LIFE = ["40% off", "Save 30%", "$50 off", "60% off", "Lifetime updates included"]
_DEAL_TERMS_LIFE = ["one-time payment", "new customers", "first 1,000 signups", "lifetime license"]
_DEAL_DISCOUNTS_GOODS = ["$70 off", "Save 20%", "Save $50", "25% off", "$30 off",
    "Buy one get one free", "Save 15%", "$100 off"]
_DEAL_TERMS_GOODS = ["while supplies last", "limit 2 per customer", "US customers",
    "in-store and online", "price valid through Sunday", "free shipping included"]
_DEAL_DISCOUNTS_FLIGHTS = ["$200 off", "Save 15%", "Free checked bag", "Round-trip from $680",
    "Save $120"]
_DEAL_TERMS_FLIGHTS = ["select dates", "round-trip economy", "advance purchase required",
    "nonstop", "US departures"]

_DEAL_BLURBS: dict[str, list[str]] = {
    "AI model subscriptions": [
        "Consumer plan with the full model, priority access at peak times, and higher usage caps.",
        "Pro tier with longer context, higher rate limits, and access to the newest features.",
        "Team-friendly plan with shared workspaces, admin controls, and centralized billing.",
    ],
    "GPU cloud rentals": [
        "On-demand GPU instances for training and inference, billed by the hour.",
        "Reserved compute with high-memory cards and fast interconnects.",
        "Serverless inference endpoints that scale to zero between requests.",
    ],
    "noise-cancelling headphones": [
        "Over-ear ANC with adaptive noise control and roughly 30-hour battery life.",
        "Wireless earbuds with strong noise cancelling and a compact charging case.",
        "Premium build, balanced sound, and multipoint Bluetooth pairing.",
    ],
    "mechanical keyboards": [
        "Hot-swappable switches, gasket mount, and a solid CNC aluminum case.",
        "Compact layout with per-key RGB and low-latency wireless.",
        "Solid build with a knob and full programmability.",
    ],
    "flights to Tokyo": [
        "Nonstop economy round trip with one free checked bag.",
        "Full-service carrier, generous baggage, and lie-flat business on long hauls.",
        "Low-cost carrier with a bare-bones fare and paid add-ons.",
    ],
    "SaaS lifetime deals": [
        "One-time payment for a perpetual individual license, no recurring fees.",
        "Stackable lifetime deal covering core features and future updates.",
        "Lifetime seat with API access and priority support.",
    ],
    "standing desks": [
        "Electric dual-motor sit-stand desk with a memory keypad and a bamboo top.",
        "Solid steel frame, high capacity, and a quiet motor.",
        "Crank-height desk with a spacious bamboo work surface.",
    ],
    "espresso machines": [
        "Semi-automatic machine with a built-in grinder and PID temperature control.",
        "Heat-exchanger boiler for back-to-back milk drinks.",
        "Compact prosumer machine with a rotary pump and a plumbed option.",
    ],
    "trail running shoes": [
        "Max-cushion trail shoe with a grippy 5mm lug outsole.",
        "Lightweight, breathable runner with a rock plate for technical terrain.",
        "Zero-drop wide-toe-box shoe with a sticky rubber outsole.",
    ],
    "VR headsets": [
        "Standalone headset with pancake lenses and full-color passthrough.",
        "Mixed-reality headset with a high-res micro-OLED display and eye tracking.",
        "PC-tethered headset with sharp panels and accurate tracking.",
    ],
}
_DEAL_BLURBS_GENERIC = ["Solid pick in this category with a strong feature set for the price.",
    "Popular choice, well reviewed, and a good value when discounted."]

_EVENT_FORMATS = ["in-person", "virtual", "hybrid"]
_EVENT_LOCATIONS = ["Vancouver, Canada", "San Francisco, CA", "New York, NY", "London, UK",
    "Berlin, Germany", "Singapore", "Tokyo, Japan", "Austin, TX", "Seattle, WA", "virtual / online"]
_EVENT_PRICES = ["$1,200", "$850 early bird", "$1,950", "free", "$2,500", "$99", "$500", "$1,500"]
_EVENT_BLURBS: dict[str, list[str]] = {
    "AI research conference": ["Annual research conference with peer-reviewed papers, workshops, and tutorials.",
        "Top-tier ML venue with poster sessions, talks, and a strong industry track."],
    "startup bootcamp": ["Intensive multi-week program with hands-on build sprints and mentor office hours.",
        "Structured curriculum, peer cohorts, and weekly pitch practice."],
    "developer meetup": ["Community meetup with technical talks, lightning demos, and networking.",
        "Monthly gathering of practitioners sharing tools, patterns, and war stories."],
    "pitch night": ["Founders pitch to a panel of investors for feedback and follow-on meetings.",
        "Fast-paced pitch event with Q&A and a community audience."],
    "hardware hackathon": ["Weekend hardware hackathon with kits, mentors, and a demo-day judging round.",
        "Build-and-ship event with component budgets and hardware labs on site."],
    "design workshop": ["Hands-on design workshop with critique rounds and portfolio reviews.",
        "Practical sessions on systems, process, and craft from working designers."],
    "robotics demo day": ["Robotics teams demo working systems to operators and investors.",
        "Live demos of manipulation, navigation, and pick-and-pack workflows."],
    "investor summit": ["Summit bringing together funds, LPs, and founders for deal-flow and panels.",
        "Curated investor gathering with deep-dive tracks and closed-door roundtables."],
    "LLM fine-tuning webinar": ["Live webinar on data prep, evals, and fine-tuning recipes for production models.",
        "Technical session covering LoRA and QLoRA, distillation, and eval harnesses."],
    "founder dinner": ["Small invited dinner for founders to swap notes and trade intros.",
        "Intimate networking dinner with a guest speaker and a roundtable discussion."],
}
_EVENT_BLURBS_GENERIC = ["Flagship gathering in this space with a strong speaker lineup."]

_STATUS_COSTS = ["$0 (no fee)", "$60k tuition", "7% equity", "0% equity, $125k investment",
    "free + $5k stipend", "$17,950 tuition", "no equity", "$500 deposit"]
_STATUS_ELIGIBILITY = ["early-stage founders", "pre-seed teams", "researchers and academics",
    "students and new grads", "underrepresented founders", "anyone building a startup",
    "teams with a working prototype"]
_STATUS_BLURBS: dict[str, list[str]] = {
    "job postings": ["Hiring across engineering, research, and product roles with remote options.",
        "Open roles with competitive comp, equity, and a strong engineering culture."],
    "grant applications": ["Non-dilutive grant for early-stage R&D with a structured application and milestones.",
        "Funding program with a written proposal, budget, and review panel."],
    "beta program access": ["Limited beta granting early access to new features and a direct line to the team.",
        "Closed beta with an NDA, feature previews, and a feedback cadence."],
    "coding bootcamp cohort": ["Immersive coding program with a structured curriculum, projects, and career support.",
        "Full-time cohort with pair programming, mock interviews, and a job guarantee."],
    "accelerator applications": ["Multi-month accelerator with funding, mentorship, partners, and a demo day.",
        "Batch program with group office hours, an alumni network, and investor intros."],
    "restaurant opening": ["New location opening with a limited menu, a soft launch, and a grand opening event.",
        "Restaurant launch with opening-week specials and a loyalty sign-up bonus."],
    "fellowship applications": ["Fellowship with a stipend, a cohort, and a structured leadership or research track.",
        "Program offering funding, mentorship, and a community of fellows."],
    "scholarship applications": ["Merit-based scholarship covering tuition, fees, and a living stipend.",
        "Award based on achievement with a one-time payment and renewal criteria."],
    "call for papers": ["Open call seeking original research papers with blind review and published proceedings.",
        "Call for papers with abstract, full-paper, and camera-ready deadlines."],
    "early-access waitlist": ["Waitlist for early access with rolling invites as capacity opens up.",
        "Early-access sign-up with priority invites and a product feedback loop."],
}
_STATUS_BLURBS_GENERIC = ["Well-regarded program in this space with a clear application process."]


def _fill_details(it: Item, rng: random.Random, topic: str) -> None:
    """Populate realistic listing details on the item, deterministic from rng."""
    if it.kind == ItemKind.DEAL:
        price, list_price = rng.choice(_DEAL_PRICES.get(topic, _DEAL_PRICES_GENERIC))
        it.price = price
        it.list_price = list_price
        # Topic-appropriate discount/terms so a hardware deal doesn't read like a
        # SaaS subscription ("annual plan" on headphones would be a giveaway).
        if topic in _DEAL_PHYSICAL_TOPICS:
            it.discount = rng.choice(_DEAL_DISCOUNTS_GOODS)
            it.terms = rng.choice(_DEAL_TERMS_GOODS)
        elif topic in _DEAL_FLIGHTS_TOPICS:
            it.discount = rng.choice(_DEAL_DISCOUNTS_FLIGHTS)
            it.terms = rng.choice(_DEAL_TERMS_FLIGHTS)
        elif topic in _DEAL_LIFETIME_TOPICS:
            it.discount = rng.choice(_DEAL_DISCOUNTS_LIFE)
            it.terms = rng.choice(_DEAL_TERMS_LIFE)
        else:  # recurring subscriptions (AI model subs, GPU cloud, generic)
            it.discount = rng.choice(_DEAL_DISCOUNTS_SUB)
            it.terms = rng.choice(_DEAL_TERMS_SUB)
        it.promo_code = rng.choice(_DEAL_PROMOS)
        it.blurb = rng.choice(_DEAL_BLURBS.get(topic, _DEAL_BLURBS_GENERIC))
    elif it.kind == ItemKind.EVENT:
        fmt = rng.choice(_EVENT_FORMATS)
        loc = rng.choice(_EVENT_LOCATIONS)
        if fmt == "virtual" and loc != "virtual / online":
            loc = "virtual / online"
        it.price = rng.choice(_EVENT_PRICES)
        it.terms = f"{fmt}, {loc}"
        it.blurb = rng.choice(_EVENT_BLURBS.get(topic, _EVENT_BLURBS_GENERIC))
        # Provenance: organizer + registration/program link. Conferences use their
        # real org + real-ish domain; other events get a generic org + eventbrite-
        # style URL. This is the "verifiable details" the agent said were missing.
        if it.entity in _CONFERENCE_ORGANIZERS:
            it.organizer = _CONFERENCE_ORGANIZERS[it.entity]
            dom = _CONFERENCE_REG_DOMAINS.get(it.entity, "events.conf")
            yr = (it.start.year if it.start else 2026)
            it.reg_url = f"https://{dom}/{yr}/registration"
        else:
            it.organizer = rng.choice(
                _EVENT_ORGANIZERS.get(topic, _EVENT_ORGANIZERS_GENERIC))
            slug = _slug(topic)
            dom = rng.choice(_EVENT_REG_DOMAINS)
            it.reg_url = f"https://{dom}/e/{slug}-{rng.randint(100, 9999)}"
    else:  # STATUS
        it.price = rng.choice(_STATUS_COSTS)
        it.terms = rng.choice(_STATUS_ELIGIBILITY)
        it.blurb = rng.choice(_STATUS_BLURBS.get(topic, _STATUS_BLURBS_GENERIC))


def _make_promo_code(entity: str, rng: random.Random, used: set[str]) -> str:
    """A product-specific, bundle-unique promo code. A real retailer code is tied
    to one product (SONY50, CHATGPT25); reusing the same few codes across
    unrelated products is a fabrication hallmark the agent flags. Derive a prefix
    from the entity's first alphabetic token, append a 2-digit number, and avoid
    collisions within the bundle."""
    toks = [t for t in re.findall(r"[A-Za-z]+", entity or "") if len(t) >= 3]
    base = (toks[0] if toks else "SAVE").upper()[:6]
    for _ in range(50):
        code = f"{base}{rng.randint(10, 60)}"
        if code not in used:
            used.add(code)
            return code
    code = f"{base}{rng.randint(100, 999)}"
    used.add(code)
    return code


def _field_names_for_kind(kind: ItemKind) -> tuple[str, ...]:
    if kind == ItemKind.DEAL:
        return ("valid_from", "expires_at")
    if kind == ItemKind.EVENT:
        # show event dates + the registration window the agent must check against T
        return ("start", "end", "window_open", "window_close")
    return ("window_open", "window_close")


_FIELD_LABELS = {
    "valid_from": "valid from", "expires_at": "expires",
    "start": "starts", "end": "ends",
    "window_open": "registration opens", "window_close": "registration closes",
}


def _date_value(item: Item, field_name: str) -> Optional[datetime]:
    return getattr(item, field_name)


# --------------------------------------------------------------------------- #
# Query kind inference + prompt
# --------------------------------------------------------------------------- #

def infer_kind(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("deal", "discount", "sale", "offer", "price", "coupon")):
        return "deal"
    if any(w in q for w in ("event", "conference", "happening", "register", "meetup", "webinar", "summit")):
        return "event"
    if any(w in q for w in ("status", "accepting", "open", "application", "hiring", "available", "current")):
        return "status"
    return "summary"


# keyword -> topic, per kind. Keeps every item in a bundle on the query's topic
# so search relevance and the task itself stay coherent.
_TOPIC_KEYWORDS = {
    "deal": [
        (("ai model", "llm", "gpt", "subscription", "api"), "AI model subscriptions"),
        (("gpu", "compute", "cloud rental"), "GPU cloud rentals"),
        (("headphone", "earbud", "audio"), "noise-cancelling headphones"),
        (("keyboard",), "mechanical keyboards"),
        (("flight", "tokyo", "travel", "airfare"), "flights to Tokyo"),
        (("saas", "software", "lifetime"), "SaaS lifetime deals"),
        (("desk",), "standing desks"),
        (("espresso", "coffee"), "espresso machines"),
        (("shoe", "running", "trail"), "trail running shoes"),
        (("vr", "headset", "vision"), "VR headsets"),
    ],
    "event": [
        (("ai research", "ai conference", "ml conference"), "AI research conference"),
        (("bootcamp",), "startup bootcamp"),
        (("meetup",), "developer meetup"),
        (("pitch",), "pitch night"),
        (("hackathon",), "hardware hackathon"),
        (("design", "workshop"), "design workshop"),
        (("robotics", "robot", "demo"), "robotics demo day"),
        (("investor", "summit"), "investor summit"),
        (("webinar", "fine-tuning", "fine tuning"), "LLM fine-tuning webinar"),
        (("founder", "dinner", "networking"), "founder dinner"),
    ],
    "status": [
        (("job", "hiring", "role"), "job postings"),
        (("grant", "funding"), "grant applications"),
        (("beta", "early access", "preview"), "beta program access"),
        (("bootcamp", "cohort"), "coding bootcamp cohort"),
        (("accelerator",), "accelerator applications"),
        (("restaurant", "opening", "cafe"), "restaurant opening"),
        (("fellowship",), "fellowship applications"),
        (("scholarship",), "scholarship applications"),
        (("call for papers", "cfp", "paper"), "call for papers"),
        (("waitlist", "early-access"), "early-access waitlist"),
    ],
}


def infer_topic(query: str, kind: str, rng: random.Random) -> str:
    q = query.lower()
    # summary tasks behave like deals (item_kind=DEAL — see _kind_to_item_kind),
    # so match against the deal keyword map so the corpus topic matches the query.
    kw_kind = "deal" if kind == "summary" else kind
    pool = {"deal": DEAL_TOPICS, "event": EVENT_TOPICS, "status": STATUS_TOPICS}.get(kw_kind, DEAL_TOPICS)
    for keywords, topic in _TOPIC_KEYWORDS.get(kw_kind, []):
        if any(k in q for k in keywords):
            return topic
    return rng.choice(pool)


def _kind_to_item_kind(kind: str) -> ItemKind:
    return {"deal": ItemKind.DEAL, "event": ItemKind.EVENT, "status": ItemKind.STATUS}.get(kind, ItemKind.DEAL)


def build_prompt(query: str, kind: str, T: datetime) -> str:
    """The prompt reads like a real user talking to a research agent: the question
    plus one natural nudge to actually search and read. We deliberately do NOT
    explain the temporality trap, hint that the time isn't given, mention stale
    sources/aggregators, or prescribe an output format. Discovering that the
    time must be fetched (get_current_time), that sources are stale, and how to
    express currency vs exclusion is the skill being trained. The tools are
    advertised to the agent via the MCP capability regardless, and the scorer
    handles free-form answers.
    """
    nudge = {
        "deal": "tell me what's actually available right now",
        "event": "tell me what's actually happening or still open right now",
        "status": "tell me what's actually open or accepting right now",
        "summary": "tell me what's actually true right now",
    }.get(kind, "tell me what's actually current right now")
    return (
        f"{query.strip()}\n\n"
        f"Search the web, read the pages you find, and {nudge}.\n\n"
        f"IMPORTANT: You MUST finish by calling the submit_answer tool. Do not write "
        f"your answer as plain text — a plain-text answer is scored zero. The ONLY "
        f"thing that is graded is the submit_answer tool call. Call submit_answer "
        f"exactly once, as your final action, with:\n"
        f"  - items: the names of the things that are available/live/open right now\n"
        f"  - supporting_doc_ids: the source-link url(s) (from search/fetch) backing "
        f"each item\n"
        f"Only items backed by a page you actually fetched count; leave out anything "
        f"expired, ended, not yet open, or undated. Even if you are unsure, you must "
        f"still call submit_answer with your best answer."
    )


# --------------------------------------------------------------------------- #
# Body rendering (template prose; --no-prose default)
# --------------------------------------------------------------------------- #

def _details_str(item: Item) -> str:
    """Realistic listing specifics: price (with pre-discount price), discount,
    promo code, and terms. Empty if the item has no details."""
    bits = []
    if item.kind == ItemKind.DEAL:
        if item.price:
            if item.list_price and item.list_price != item.price:
                bits.append(f"{item.price} (was {item.list_price})")
            else:
                bits.append(item.price)
        if item.discount:
            bits.append(item.discount)
        if item.promo_code:
            bits.append(f"code {item.promo_code}")
        if item.terms:
            bits.append(item.terms)
    else:
        # events: "price · format, location · organizer X · register: url";
        # status: "cost · eligibility"
        if item.price:
            bits.append(item.price)
        if item.terms:
            bits.append(item.terms)
        if item.kind == ItemKind.EVENT:
            if item.organizer:
                bits.append(f"hosted by {item.organizer}")
            if item.reg_url:
                bits.append(f"register: {item.reg_url}")
    return ", ".join(bits)


def _item_line(item: Item, di: DocItem, T: datetime, rng: random.Random) -> str:
    """One rendered line for an item within a doc, using that doc's label forms."""
    parts = [item.name]
    if item.currency_phrase:
        parts.append(f"({item.currency_phrase})")
    details = _details_str(item)
    if details:
        parts.append("— " + details)
    date_bits = []
    for f in di.show_fields:
        label = _FIELD_LABELS.get(f, f.replace('_', ' '))
        if f in di.override_labels:
            date_bits.append(f"{label}: {di.override_labels[f]}")
            continue
        val = _date_value(item, f)
        lbl = render_label(val, T, di.label_forms.get(f, LabelForm.MISSING), rng)
        if lbl:
            date_bits.append(f"{label}: {lbl}")
    if date_bits:
        parts.append("[" + ", ".join(date_bits) + "]")
    return " ".join(parts)


def render_body(doc: Doc, bundle_items: dict[str, Item], T: datetime, rng: random.Random) -> str:
    items = [bundle_items[di.item_id] for di in doc.items if di.item_id in bundle_items]
    st = doc.source_type

    if st == "aggregator_list":
        lines = [f"# {doc.title}", ""]
        if doc.published_label:
            lines.append(f"_(last updated {doc.published_label})_")
        lines.append(f"_{len(items)} picks, all current as of posting. Verify before buying._")
        lines.append("")
        for i, (it, di) in enumerate(zip(items, doc.items), 1):
            lines.append(f"### {i}. {_item_line(it, di, T, rng)}")
            if it.blurb:
                lines.append("")
                lines.append(it.blurb)
            lines.append("")
        return "\n".join(lines)

    if st == "social_post":
        sub = doc.source_subtype or "tweet"
        lines = []
        if sub == "reddit":
            lines.append(f"r/deals • Posted by u/{doc.engagement.get('user','anon')} "
                         f"{doc.published_label or '(no date)'}")
        elif sub == "hackernews":
            lines.append(f"Hacker News • {doc.engagement.get('user','anon')} "
                         f"{doc.published_label or '(no date)'}")
        elif sub == "linkedin":
            lines.append(f"{doc.engagement.get('user','Anon')} • LinkedIn "
                         f"{doc.published_label or '(no date)'}")
        else:
            lines.append(f"@{doc.engagement.get('user','anon')} "
                         f"{doc.published_label or ''}")
        if doc.edited:
            lines.append("_(post edited)_")
        lines.append("")
        lines.append(doc.title)
        for it, di in zip(items, doc.items):
            lines.append(_item_line(it, di, T, rng))
        if doc.engagement:
            eng = []
            if "upvotes" in doc.engagement:
                eng.append(f"{doc.engagement['upvotes']} upvotes")
            if "likes" in doc.engagement:
                eng.append(f"{doc.engagement['likes']} likes")
            if eng:
                lines.append(f"_{', '.join(eng)}_")
        if doc.replies:
            lines.append("")
            lines.append("Replies:")
            for r in doc.replies:
                tag = " [correction]" if r.is_correction else ""
                lines.append(f"> @{r.user}{tag}: {r.text}")
        return "\n".join(lines)

    if st == "news":
        lines = [doc.title, ""]
        if doc.published_label:
            lines.append(f"By Staff Reporter | {doc.published_label}")
        lines.append("")
        for it, di in zip(items, doc.items):
            lines.append(f"- {_item_line(it, di, T, rng)}")
            if it.blurb:
                lines.append(f"  {it.blurb}")
        return "\n".join(lines)

    if st == "marketing":
        lines = [f"# {doc.title}", ""]
        if doc.published_label:
            lines.append(f"(page updated {doc.published_label})")
        lines.append("## Limited-time offers")
        for it, di in zip(items, doc.items):
            lines.append(f"- {_item_line(it, di, T, rng)}")
            if it.blurb:
                lines.append(f"  {it.blurb}")
        return "\n".join(lines)

    if st == "pdf_receipt":
        lines = ["===== RECEIPT / ORDER CONFIRMATION =====", ""]
        if doc.published_label:
            lines.append(f"Date: {doc.published_label}")
        lines.append(f"Order: {doc.title}")
        lines.append("")
        for it, di in zip(items, doc.items):
            lines.append(f"* {_item_line(it, di, T, rng)}")
        return "\n".join(lines)

    # forum (reddit-ish thread)
    lines = [f"r/{doc.engagement.get('sub','topics')} • {doc.published_label or '(no date)'}", doc.title, ""]
    for it, di in zip(items, doc.items):
        lines.append(f"- {_item_line(it, di, T, rng)}")
    return "\n".join(lines)


def _make_snippet(doc: Doc, bundle_items: dict[str, Item], T: datetime, rng: random.Random) -> str:
    """Short search-result snippet: title + first item line + published label."""
    bits = [doc.title]
    if doc.items:
        di = doc.items[0]
        it = bundle_items.get(di.item_id)
        if it:
            bits.append(_item_line(it, di, T, rng))
    if doc.published_label:
        bits.append(f"posted {doc.published_label}")
    return " — ".join(bits)


# --------------------------------------------------------------------------- #
# Realistic URLs per source type (real, well-known domains). The agent in eval
# was refusing to read pages whose hostnames looked like placeholders
# (``example.com`` / ``blog-aggregator``), so we mint real-looking URLs. Paths
# stay deterministic from doc_idx so URLs are unique and reproducible per seed.
# --------------------------------------------------------------------------- #

_DOMAINS = {
    "news": ["techcrunch.com", "arstechnica.com", "theverge.com", "zdnet.com",
             "wired.com", "cnet.com", "reuters.com"],
    "forum": ["lobste.rs", "news.ycombinator.com", "discussions.apple.com",
              "community.openai.com", "forums.macrumors.com"],
    "marketing": ["blog.google", "blogs.microsoft.com", "aws.amazon.com/blogs",
                  "blog.huggingface.co", "medium.com"],
    "pdf_receipt": ["drive.google.com", "assets.acme.io", "s3.amazonaws.com"],
    "aggregator_list": ["www.pcmag.com", "www.tomsguide.com", "www.zdnet.com",
                        "www.techradar.com", "www.cnet.com", "www.g2.com",
                        "zapier.com", "www.producthunt.com"],
}

_REDDIT_SUBS = ["technology", "artificial", "MachineLearning", "deals",
                "apple", "google", "tech", "programming"]

# Aggregator URL path + slug templates — varied per doc so multiple aggregators
# on the same domain get distinct, natural-looking URLs (real outlets rotate
# these: /picks/the-best-X, /reviews/X, /best-picks/X, /article/the-best-X).
_AGG_PATHS = ["picks", "reviews", "best-picks", "article/picks", "roundup"]
_AGG_SLUGS = ["the-best-{s}", "best-{s}-this-year", "top-{s}-right-now",
              "the-best-{s}-deals", "best-{s}-for-the-money"]
# Natural per-article slug variety for news/forum and marketing docs, so two
# docs of the same type+topic get distinct URLs (no naked numeric article-id).
_NEWS_SLUGS = ["{s}", "{s}-update", "the-latest-on-{s}", "{s}-this-week",
               "whats-new-in-{s}", "{s}-news"]
_MARKETING_SLUGS = ["blog/{s}", "blog/{s}-announcement", "blog/introducing-{s}",
                    "blog/the-future-of-{s}", "blog/{s}-update", "blog/{s}-launch"]


def _slug(topic: str) -> str:
    s = "".join(c if c.isalnum() else "-" for c in (topic or "").lower())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "topic"


def _realistic_url(source_type: str, subtype: str, topic: str,
                   doc_idx: int, rng: random.Random,
                   used: set[str] | None = None) -> str:
    """A real-looking URL for a doc, deterministic from doc_idx (+ rng for domain
    choice) and **unique within the bundle**. Editorial URLs use natural article
    slugs (no trailing numeric article-id, which was a fabrication tell:
    ``pcmag.com/picks/best-X-148``); social URLs keep realistic platform ids
    (reddit/HN/x have numeric ids). When two docs of the same type+topic would
    collide (e.g. two marketing posts → ``blogs.microsoft.com/blog/X``), the
    slug template + domain are rotated deterministically until the URL is new —
    so URLs stay natural AND never duplicate (duplicates broke fetch-by-url and
    collapsed the grader's per-doc name map)."""
    slug = _slug(topic)
    n = doc_idx * 499 + 1200  # deterministic fake status/comment id
    used = used if used is not None else set()

    def _claim(u: str) -> str:
        # deterministic uniqueness: rotate (template, domain) until unused
        if u not in used:
            used.add(u)
            return u
        return u  # placeholder; caller path below handles rotation

    if source_type == "social_post":
        if subtype == "tweet":
            u = f"https://x.com/u_{hashlib.sha1(f'tw{doc_idx}'.encode()).hexdigest()[:8]}/status/{n}"
        elif subtype == "reddit":
            sub = rng.choice(_REDDIT_SUBS)
            u = f"https://www.reddit.com/r/{sub}/comments/{n}/{slug}/"
        elif subtype == "hackernews":
            u = f"https://news.ycombinator.com/item?id={doc_idx * 73 + 100}"
        elif subtype == "linkedin":
            u = f"https://www.linkedin.com/posts/u_{hashlib.sha1(f'li{doc_idx}'.encode()).hexdigest()[:8]}_{n}"
        else:
            u = f"https://x.com/u_{hashlib.sha1(f'soc{doc_idx}'.encode()).hexdigest()[:8]}/status/{n}"
        # social ids are unique per doc_idx; still record to prevent any clash
        used.add(u)
        return u

    if source_type == "aggregator_list":
        doms = _DOMAINS["aggregator_list"]
        paths = _AGG_PATHS
        tmpls = _AGG_SLUGS
        cand = f"https://{rng.choice(doms)}/{paths[doc_idx % len(paths)]}/{tmpls[doc_idx % len(tmpls)].format(s=slug)}"
        if cand not in used:
            used.add(cand)
            return cand
        # rotate deterministically through (path, tmpl, dom) until unique
        for a in range(len(paths) * len(tmpls) * len(doms)):
            p = paths[(doc_idx + a) % len(paths)]
            t = tmpls[(doc_idx + a) % len(tmpls)]
            dom = doms[(doc_idx + a) % len(doms)]
            c = f"https://{dom}/{p}/{t.format(s=slug)}"
            if c not in used:
                used.add(c)
                return c
        used.add(cand)
        return cand

    if source_type in _DOMAINS:
        doms = _DOMAINS[source_type]
        if source_type == "pdf_receipt":
            u = f"https://{rng.choice(doms)}/receipts/{slug}-{1000 + doc_idx * 37}.pdf"
            used.add(u)
            return u
        if source_type == "marketing":
            pool = _MARKETING_SLUGS
        else:  # news / forum
            pool = _NEWS_SLUGS
        cand = f"https://{rng.choice(doms)}/{pool[doc_idx % len(pool)].format(s=slug)}"
        if cand not in used:
            used.add(cand)
            return cand
        for a in range(len(pool) * len(doms)):
            dom = doms[(doc_idx + a) % len(doms)]
            c = f"https://{dom}/{pool[(doc_idx + a) % len(pool)].format(s=slug)}"
            if c not in used:
                used.add(c)
                return c
        used.add(cand)
        return cand
    u = f"https://example.com/{slug}-{doc_idx}"
    used.add(u)
    return u


# --------------------------------------------------------------------------- #
# Optional LLM proseify hook (code owns truth; LLM only rephrases surface)
# --------------------------------------------------------------------------- #

def _ollama_client():
    """OpenAI-compatible client pointed at a local Ollama server."""
    from openai import OpenAI
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return OpenAI(base_url=base_url, api_key="ollama")  # api_key ignored by Ollama


def _prose_cache_dir() -> Path:
    p = Path(os.environ.get("TEMPORA_PROSE_CACHE",
                            str(Path.home() / ".cache" / "tempora_prose")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cached_or_generate(model: str, skeleton: str, client) -> str:
    """Rewrite one doc's template prose via Ollama, with an on-disk cache keyed by
    (model, skeleton) so repeated runs are free and generation stays deterministic."""
    key = hashlib.sha256(f"{model}\x00{skeleton}".encode()).hexdigest()
    cache_file = _prose_cache_dir() / f"{key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    system = (
        "You rewrite web pages into natural, realistic prose. HARD RULES: keep EVERY "
        "date, name, number, list item, heading, and the source format (news article, "
        "Reddit/HN/LinkedIn post, marketing page, receipt, or ranked list) exactly as "
        "given. Do NOT add or remove any item or any date. Do NOT 'correct' or update "
        "dates even if they look stale — staleness is intentional. Preserve currency "
        "phrases ('right now', 'still live', 'coming soon') and any reply threads. "
        "Output ONLY the rewritten page, no preamble, no commentary, no code fences."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": skeleton}],
        temperature=0.7,
    )
    text = (resp.choices[0].message.content or "").strip()
    # strip accidental surrounding code fences
    if text.startswith("```"):
        text = text.strip("`").lstrip("\n").strip()
        if text.lower().startswith("markdown"):
            text = text.split("\n", 1)[1] if "\n" in text else text
    cache_file.write_text(text, encoding="utf-8")
    return text


def proseify(bundle: DocBundle, model: Optional[str] = None) -> None:
    """Optional: rewrite each doc.body in richer prose via a local Ollama model.

    The skeleton + ground truth are already fixed by ``build_bundle``; this only
    changes surface wording, and results are cached by (model, skeleton) so it is
    deterministic and cheap to re-run. Default model is the smallest local model
    pulled on this machine; override with ``model=`` or the ``OLLAMA_MODEL`` env
    var. Not used in --no-prose mode (the default).
    """
    model = model or os.environ.get("OLLAMA_MODEL", "glm-5.2:cloud")
    try:
        client = _ollama_client()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("proseify requires the `openai` package") from e
    for doc in bundle.docs:
        try:
            doc.body = _cached_or_generate(model, doc.body, client)
        except Exception as e:
            # Keep the deterministic template prose rather than crashing the env.
            import sys
            print(f"[proseify] WARN: Ollama call failed for {doc.doc_id} "
                  f"({model}); keeping template prose. {e}", file=sys.stderr)
            break


# --------------------------------------------------------------------------- #
# Bundle assembly
# --------------------------------------------------------------------------- #

def _sample_lure_composition(rng: random.Random, n_lures: int) -> dict[str, int]:
    """Distribute lures across expired / future / undated with variety per bundle."""
    if n_lures == 0:
        return {"expired": 0, "future": 0, "undated_ambiguous": 0}
    # pick a profile
    profile = rng.choices(
        ["mixed", "mostly_expired", "mostly_future", "mostly_undated"],
        weights=[0.4, 0.25, 0.2, 0.15], k=1)[0]
    weights = {
        "mixed": [0.4, 0.3, 0.3],
        "mostly_expired": [0.75, 0.1, 0.15],
        "mostly_future": [0.1, 0.75, 0.15],
        "mostly_undated": [0.1, 0.1, 0.8],
    }[profile]
    counts = {"expired": 0, "future": 0, "undated_ambiguous": 0}
    keys = ["expired", "future", "undated_ambiguous"]
    for _ in range(n_lures):
        k = rng.choices(keys, weights=weights, k=1)[0]
        counts[k] += 1
    return counts


def build_bundle(
    query: str,
    seed: int,
    difficulty: float = 0.3,
    *,
    use_llm: bool = False,
    now_override: Optional[datetime] = None,
    force: Optional[dict] = None,
) -> DocBundle:
    """Generate one DocBundle deterministically from (query, seed, difficulty).

    ``force`` is an optional dict for testing/audit, e.g.
    ``{"aggregator": True, "social": True, "thread_correction": True}``.
    """
    rng = random.Random(_seed_hash(query, seed))
    force = force or {}

    T = now_override or (BASE_NOW + timedelta(days=rng.randint(-160, 160),
                                               hours=rng.randint(-10, 10)))
    kind = infer_kind(query)
    item_kind = _kind_to_item_kind(kind)
    topic = infer_topic(query, kind, rng)

    # ---- shape: counts ----
    # Larger corpus than a single search page: search() returns a ranked top-10
    # (Exa default), so with 12-20 docs the agent sees a subset and must choose
    # what to fetch — gold can be buried on "page 2". Items capped at 15 to stay
    # within the unique entity pools (avoids duplicate-name matching noise).
    n_docs = rng.randint(12, 20)
    n_items = rng.randint(8, 15)

    # ---- gold / lure ratio (1:4 .. 4:1) ----
    gold_frac = rng.uniform(0.2, 0.8)
    n_gold = max(1, round(n_items * gold_frac))
    n_lures = max(0, n_items - n_gold)
    comp = _sample_lure_composition(rng, n_lures)

    # ---- build canonical items with target states ----
    items: list[Item] = []
    targets: list[ItemState] = (
        [ItemState.ACTIVE] * n_gold
        + [ItemState.EXPIRED] * comp["expired"]
        + [ItemState.FUTURE] * comp["future"]
        + [ItemState.UNDATED_AMBIGUOUS] * comp["undated_ambiguous"]
    )
    rng.shuffle(targets)
    # Topic-aware entity pool: prefer real, topic-specific brand names so the
    # corpus reads as genuine (ChatGPT Plus, Sony WH-1000XM5, NeurIPS, ...).
    # Fall back to the per-kind generic real-brand pool if a topic is uncovered.
    entity_pool = TOPIC_ENTITIES.get(topic) or {
        ItemKind.DEAL: DEAL_ENTITIES,
        ItemKind.EVENT: EVENT_ENTITIES,
        ItemKind.STATUS: STATUS_ENTITIES,
    }[item_kind]
    # Sample without replacement so every item name in the bundle is unique.
    if n_items <= len(entity_pool):
        entities = rng.sample(entity_pool, n_items)
    else:
        entities = rng.sample(entity_pool, len(entity_pool)) + [
            f"{e} #{i}" for i, e in enumerate(rng.choices(entity_pool, k=n_items - len(entity_pool)), 2)]
    used_codes: set[str] = set()
    used_urls: set[str] = set()   # guarantees every doc URL is unique in the bundle
    for idx, tgt in enumerate(targets):
        it = _make_item(rng, idx, item_kind, tgt, T, topic, entities[idx])
        it.state = compute_state(it, T)
        _fill_details(it, rng, topic)
        if it.kind == ItemKind.DEAL:
            it.promo_code = _make_promo_code(entities[idx], rng, used_codes)
        items.append(it)
    by_id = {it.id: it for it in items}

    # ---- decide source types for docs ----
    # Real topics have several "best of" blogs and multiple social posts, not
    # just 0-1 of each. Sample a few of each, then pad with primaries (many will
    # be noise docs with 0 relevant items — realistic).
    has_aggregator = force.get("aggregator", rng.random() < 0.8)
    has_social = force.get("social", rng.random() < 0.85)

    n_agg = force.get("n_aggregator", rng.randint(1, 3)) if has_aggregator else 0
    n_social = force.get("n_social", rng.randint(1, 4)) if has_social else 0

    source_plan: list[str] = (
        ["aggregator_list"] * n_agg + ["social_post"] * n_social
    )
    primary_pool = ["news", "forum", "marketing", "pdf_receipt"]
    while len(source_plan) < n_docs:
        source_plan.append(rng.choice(primary_pool))
    rng.shuffle(source_plan)
    source_plan = source_plan[:n_docs]

    # ---- assign items to docs ----
    # Social gets 1 (OP), aggregator gets a capped slice, primaries get 0..4.
    # Build social FIRST so it never starves; cap the aggregator so items stay
    # distributed across docs (no single doc consumes the whole bundle).
    docs: list[Doc] = []
    doc_idx = 0
    item_queue = list(items)
    rng.shuffle(item_queue)
    n_primary_docs = sum(1 for s in source_plan if s not in ("aggregator_list", "social_post"))

    def take(n: int) -> list[Item]:
        out = item_queue[:n]
        del item_queue[:n]
        return out

    # Per-(doc,item) label forms + conflict overrides
    def make_doc_item(it: Item, *, force_missing: bool = False,
                      override: Optional[dict] = None) -> DocItem:
        fields = _field_names_for_kind(it.kind)
        # choose how many fields to show (0..all) — fewer at higher difficulty
        show_n = len(fields)
        if not force_missing and it.state != ItemState.UNDATED_AMBIGUOUS:
            # sometimes hide a field
            if rng.random() < 0.25 + 0.35 * difficulty:
                show_n = max(0, show_n - rng.randint(1, len(fields)))
        show = fields[:show_n]
        label_forms = {f: _pick_label_form(rng, difficulty) for f in show}
        return DocItem(item_id=it.id, show_fields=tuple(show), label_forms=label_forms,
                       override_labels=override or {})

    # Social docs (built first). Multiple posts; each picks an OP item from the
    # full item list (with replacement — realistic: many people post about the
    # same hot deal). Social does NOT consume from item_queue, so aggregators and
    # primaries keep the full pool.
    for _ in range(n_social):
        if not items:
            break
        sub = rng.choice(SOCIAL_SUBTYPES)
        op_items = [rng.choice(items)]
        dis = [make_doc_item(it) for it in op_items]
        # social often undated or relative-only
        if rng.random() < 0.4:
            dis = [make_doc_item(it, force_missing=True) for it in op_items]
        soc_pub = None
        if rng.random() < 0.6:
            soc_pub = T - timedelta(days=rng.randint(0, 30))
        lf = _pick_label_form(rng, difficulty, prefer_missing=(soc_pub is None))
        pub_lbl = render_label(soc_pub, T, lf, rng) if soc_pub else ""
        # thread correction: a reply that flips the claim (prose only; truth unchanged)
        replies = []
        want_correction = force.get("thread_correction", rng.random() < 0.4)
        if want_correction and op_items:
            op = op_items[0]
            flip = "actually expired / ended" if op.state == ItemState.ACTIVE else "actually still live"
            replies.append(Reply(user=rng.choice(["dev42", "sara_k", "nomad", "paulb"]),
                                  text=f"Update: this is {flip} as of recently.",
                                  is_correction=True))
        # engagement decoys
        engagement = {"user": rng.choice(["dev42", "sara_k", "nomad", "paulb", "alex_r"])}
        if sub == "reddit":
            engagement["upvotes"] = rng.randint(3, 4200)
            engagement["sub"] = rng.choice(["deals", "MachineLearning", "startups", "tech"])
        elif sub == "hackernews":
            engagement["upvotes"] = rng.randint(5, 800)
        elif sub == "linkedin":
            engagement["likes"] = rng.randint(2, 300)
        else:
            engagement["likes"] = rng.randint(1, 5000)
        title = (op_items[0].name + " " + op_items[0].currency_phrase).strip()
        d = Doc(doc_id=f"doc-{doc_idx}",
                url=_realistic_url("social_post", sub, topic, doc_idx, rng, used_urls),
                title=title, source_type="social_post", source_subtype=sub,
                published_at=soc_pub, published_label_form=lf, published_label=pub_lbl,
                items=dis, replies=replies, engagement=engagement,
                edited=rng.random() < 0.2)
        docs.append(d)
        doc_idx += 1

    # Aggregator docs (several "best of" blogs). Each takes a fair slice of the
    # remaining item_queue, reserving items for later aggregators and primaries.
    for i in range(n_agg):
        remaining = len(item_queue)
        if remaining <= 0:
            break
        reserve = (n_agg - i - 1) + (1 if n_primary_docs else 0)
        n_take = min(rng.randint(3, 8), max(1, remaining - reserve))
        agg_items = take(n_take)
        if not agg_items:
            break
        dis = [make_doc_item(it) for it in agg_items]
        # aggregator is usually STALE relative to T (its "right now" is old)
        agg_pub = T - timedelta(days=rng.randint(30, 180))
        lf = _pick_label_form(rng, difficulty, prefer_missing=False)
        pub_lbl = render_label(agg_pub, T, lf, rng)
        title = f"Top {len(agg_items)} {topic} picks right now"
        d = Doc(doc_id=f"doc-{doc_idx}",
                url=_realistic_url("aggregator_list", "", topic, doc_idx, rng, used_urls),
                title=title, source_type="aggregator_list",
                published_at=agg_pub, published_label_form=lf, published_label=pub_lbl,
                items=dis)
        docs.append(d)
        doc_idx += 1

    # Primary docs (news/forum/marketing/pdf_receipt). Allow noise docs (0 items).
    primary_types = [s for s in source_plan if s not in ("aggregator_list", "social_post")]
    for st in primary_types:
        cap = rng.randint(0, 4)
        doc_items = take(cap)
        dis = [make_doc_item(it) for it in doc_items]
        pub = None
        if rng.random() < 0.7:
            pub = T - timedelta(days=rng.randint(1, 200))
        lf = _pick_label_form(rng, difficulty, prefer_missing=(pub is None and rng.random() < 0.5))
        pub_lbl = render_label(pub, T, lf, rng) if pub else ""
        title = (f"{rng.choice(['Update','News','Report','Notice'])}: {topic}")
        if not doc_items:
            title = f"Background: {topic} (no specific listings)"
        d = Doc(doc_id=f"doc-{doc_idx}",
                url=_realistic_url(st, "", topic, doc_idx, rng, used_urls),
                title=title, source_type=st,
                published_at=pub, published_label_form=lf, published_label=pub_lbl,
                items=dis)
        docs.append(d)
        doc_idx += 1

    # ---- conflicts: re-feature a few items in a second doc with a wrong date ----
    n_conflicts = 0
    if rng.random() < 0.3 + 0.4 * difficulty:
        n_conflicts = rng.randint(1, 2)
    featured = {di.item_id for d in docs for di in d.items}
    eligible = [it for it in items if it.id in featured and it.kind == item_kind
                and it.state in (ItemState.EXPIRED, ItemState.FUTURE, ItemState.ACTIVE)]
    rng.shuffle(eligible)
    for it in eligible[:n_conflicts]:
        # pick a doc that already contains it, add an override conflict entry elsewhere
        fields = _field_names_for_kind(it.kind)
        # fabricate a wrong date that implies the opposite state
        if it.state == ItemState.EXPIRED:
            wrong = explicit_phrase(T + timedelta(days=rng.randint(10, 90)))
        elif it.state == ItemState.FUTURE:
            wrong = explicit_phrase(T - timedelta(days=rng.randint(10, 90)))
        else:
            wrong = explicit_phrase(T + timedelta(days=rng.randint(5, 30)))
        override = {fields[-1]: wrong}  # override the "end"/"expiry"/"close" field
        # attach to a random primary doc that does NOT already show this item
        candidates = [d for d in docs if d.source_type in ("news", "forum", "marketing")
                      and all(di.item_id != it.id for di in d.items)]
        if candidates:
            host = rng.choice(candidates)
            di = DocItem(item_id=it.id, show_fields=(fields[-1],),
                         label_forms={fields[-1]: LabelForm.EXPLICIT},
                         override_labels=override)
            host.items.append(di)

    # ---- search rank hints: lures boosted, gold sunk at higher difficulty ----
    for d in docs:
        base = 0.4 + rng.random() * 0.2
        lure_score = 0.0
        gold_score = 0.0
        for di in d.items:
            it = by_id.get(di.item_id)
            if it and it.state != ItemState.ACTIVE:
                lure_score = max(lure_score, 0.5)
            elif it and it.state == ItemState.ACTIVE:
                gold_score = max(gold_score, 0.5)
        if d.source_type == "aggregator_list":
            base += 0.15  # SEO dominance
        rank = base + difficulty * lure_score - difficulty * 0.5 * gold_score
        d.search_rank_hint = max(0.01, min(0.99, rank))

    # Guarantee at least one gold-bearing doc is visible on the default search
    # page (top SEARCH_LIMIT). With a larger corpus, gold can otherwise be buried
    # below the fold and the task becomes unsolvable. Only the single best gold
    # doc is surfaced to the boundary; the rest may stay buried (the agent finds
    # a lead, fetches, reasons — and can request more numResults to dig deeper).
    SEARCH_LIMIT = 10
    if len(docs) > SEARCH_LIMIT:
        ranked = sorted(docs, key=lambda d: d.search_rank_hint, reverse=True)
        gold_docs = [d for d in ranked
                     if any(by_id.get(di.item_id) and by_id[di.item_id].state == ItemState.ACTIVE
                            for di in d.items)]
        if gold_docs and ranked.index(gold_docs[0]) >= SEARCH_LIMIT:
            cutoff = ranked[SEARCH_LIMIT - 1]
            gold_docs[0].search_rank_hint = min(0.99, cutoff.search_rank_hint + 0.001)

    # ---- render bodies + snippets ----
    for d in docs:
        d.body = render_body(d, by_id, T, rng)
        d.snippet = _make_snippet(d, by_id, T, rng)

    gold_ids = [it.id for it in items if it.state == ItemState.ACTIVE]
    lure_ids = [it.id for it in items if it.state != ItemState.ACTIVE]

    prompt = build_prompt(query, kind, T)

    bundle = DocBundle(now=T, prompt=prompt, query=query, query_kind=kind,
                       docs=docs, items=items, gold_ids=gold_ids, lure_ids=lure_ids)

    if use_llm:
        proseify(bundle)

    return bundle


# --------------------------------------------------------------------------- #
# Structural fingerprint + variety audit
# --------------------------------------------------------------------------- #

def fingerprint(b: DocBundle) -> tuple:
    """A coarse structural signature for variety auditing (ignores prose)."""
    n_lures_exp = sum(1 for it in b.items if it.state == ItemState.EXPIRED)
    n_lures_fut = sum(1 for it in b.items if it.state == ItemState.FUTURE)
    n_lures_und = sum(1 for it in b.items if it.state == ItemState.UNDATED_AMBIGUOUS)
    # date-label histogram over displayed doc dates
    hist = {"explicit": 0, "relative": 0, "vague": 0, "missing": 0}
    n_conflicts = 0
    for d in b.docs:
        hist[d.published_label_form.value] = hist.get(d.published_label_form.value, 0) + 1
        for di in d.items:
            for f, lf in di.label_forms.items():
                hist[lf.value] = hist.get(lf.value, 0) + 1
            if di.override_labels:
                n_conflicts += 1
    has_agg = any(d.source_type == "aggregator_list" for d in b.docs)
    has_social = any(d.source_type == "social_post" for d in b.docs)
    has_correction = any(r.is_correction for d in b.docs for r in d.replies)
    top_hit_state = "none"
    if b.docs:
        ranked = sorted(b.docs, key=lambda d: d.search_rank_hint, reverse=True)
        top = ranked[0]
        states = [b.item_by_id(di.item_id).state.value for di in top.items
                  if b.item_by_id(di.item_id)]
        top_hit_state = states[0] if states else "empty"
    ratio = round(len(b.gold_ids) / max(1, len(b.lure_ids)), 1)
    return (len(b.docs), len(b.items), n_lures_exp, n_lures_fut, n_lures_und,
            tuple(sorted(hist.items())), n_conflicts, ratio, top_hit_state,
            has_agg, has_social, has_correction)


def audit(n_seeds: int = 500, difficulty: float = 0.3, query: str = "deals on AI models right now") -> dict:
    from collections import Counter
    fps = []
    for s in range(n_seeds):
        fps.append(fingerprint(build_bundle(query, s, difficulty)))
    c = Counter(fps)
    distinct = len(c)
    total = len(fps)
    probs = [v / total for v in c.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_share = max(probs)
    has_agg = sum(1 for f in fps if f[9])
    has_social = sum(1 for f in fps if f[10])
    has_correction = sum(1 for f in fps if f[11])
    return {
        "n_seeds": n_seeds,
        "distinct_fingerprints": distinct,
        "entropy_bits": round(entropy, 2),
        "max_fingerprint_share": round(max_share, 3),
        "top5_fingerprints": c.most_common(5),
        "bundles_with_aggregator": has_agg,
        "bundles_with_social": has_social,
        "bundles_with_thread_correction": has_correction,
    }


# --------------------------------------------------------------------------- #
# Serialization (for dumps/debug)
# --------------------------------------------------------------------------- #

def _bundle_to_dict(b: DocBundle) -> dict:
    return {
        "now": b.now.isoformat(),
        "query": b.query,
        "query_kind": b.query_kind,
        "prompt": b.prompt,
        "gold_ids": b.gold_ids,
        "lure_ids": b.lure_ids,
        "items": [
            {"id": it.id, "kind": it.kind.value, "name": it.name, "entity": it.entity,
             "state": it.state.value,
             "valid_from": it.valid_from and it.valid_from.isoformat(),
             "expires_at": it.expires_at and it.expires_at.isoformat(),
             "start": it.start and it.start.isoformat(),
             "end": it.end and it.end.isoformat(),
             "window_open": it.window_open and it.window_open.isoformat(),
             "window_close": it.window_close and it.window_close.isoformat()}
            for it in b.items
        ],
        "docs": [
            {"doc_id": d.doc_id, "url": d.url, "title": d.title,
             "source_type": d.source_type, "source_subtype": d.source_subtype,
             "published_at": d.published_at and d.published_at.isoformat(),
             "published_label": d.published_label,
             "search_rank_hint": d.search_rank_hint, "edited": d.edited,
             "engagement": d.engagement,
             "items": [{"item_id": di.item_id, "show_fields": list(di.show_fields),
                        "label_forms": {k: v.value for k, v in di.label_forms.items()},
                        "override_labels": di.override_labels} for di in d.items],
             "replies": [{"user": r.user, "text": r.text, "is_correction": r.is_correction}
                         for r in d.replies],
             "body": d.body}
            for d in b.docs
        ],
    }


# --------------------------------------------------------------------------- #
# CLI: determinism check, variety audit, sample dump
# --------------------------------------------------------------------------- #

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Tempora generator")
    ap.add_argument("--query", default="deals on AI models right now")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--use-llm", action="store_true",
                    help="Rewrite doc prose via Ollama (default off; needs Ollama running)")
    ap.add_argument("--ollama-model", default=None,
                    help="Ollama model for --use-llm (default: OLLAMA_MODEL env or glm-5.2:cloud)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("check")
    sub.add_parser("audit")
    sub.add_parser("dump")
    args = ap.parse_args()
    cmd = args.cmd or "check"

    if cmd == "audit":
        print(json.dumps(audit(500, args.difficulty, args.query), indent=2))
        return

    if cmd == "dump":
        b = build_bundle(args.query, args.seed, args.difficulty, use_llm=False)
        if args.use_llm:
            proseify(b, model=args.ollama_model)
        print(b.to_json())
        return

    # determinism check
    b1 = build_bundle(args.query, args.seed, args.difficulty, use_llm=False)
    b2 = build_bundle(args.query, args.seed, args.difficulty, use_llm=False)
    same = (b1.to_json() == b2.to_json())
    print(f"query={args.query!r} seed={args.seed} difficulty={args.difficulty}")
    print(f"docs={len(b1.docs)} items={len(b1.items)} "
          f"gold={len(b1.gold_ids)} lure={len(b1.lure_ids)}")
    print(f"gold_ids={b1.gold_ids}  lure_states="
          f"{[b1.item_by_id(i).state.value for i in b1.lure_ids]}")
    print(f"deterministic (identical on re-run): {same}")
    print()
    print("Fingerprint:", fingerprint(b1))
    print()
    print("=== prompt ===")
    print(b1.prompt)
    print()
    print("=== docs (ranked as search would order) ===")
    for d in sorted(b1.docs, key=lambda x: x.search_rank_hint, reverse=True):
        print(f"\n[{d.doc_id}] rank={d.search_rank_hint:.2f} "
              f"type={d.source_type} pub={d.published_label or '(no date)'}")
        print(f"title: {d.title}")
        print(d.body)

    if args.use_llm:
        import os as _os
        model = args.ollama_model or _os.environ.get("OLLAMA_MODEL", "glm-5.2:cloud")
        print(f"\n=== Ollama prose (model={model}) ===")
        bprose = build_bundle(args.query, args.seed, args.difficulty, use_llm=False)
        proseify(bprose, model=model)
        for d in sorted(bprose.docs, key=lambda x: x.search_rank_hint, reverse=True)[:2]:
            print(f"\n[{d.doc_id}] type={d.source_type}")
            print(d.body)


if __name__ == "__main__":
    _cli()