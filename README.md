# Ady Evolve — a temporality-reasoning RL environment for deep-research agents

**HUD × YC Frontier RL Environments Hackathon, June 20–21 2026.**

Ady Evolve (a.k.a. Tempora) trains and tests whether a deep-research agent can reason about
**temporality** across heterogeneous web sources relative to a reference time
`T` (only knowable by calling `get_current_time`) — and report only what is
verifiably current, excluding expired deals, past events, not-yet-open items,
stale "right now" aggregator lists, and undated/unconfirmable social posts.

## The skill it exercises

A real research agent doesn't just retrieve — it has to decide what is *still
true* from a mix of:

- **dated primary pages** (explicit ISO dates),
- **relative-dated pages** ("3 days ago", "last week" — anchored to `T`),
- **vague / undated pages** ("this season", no date),
- **aggregator lists** titled "best X right now" whose "right now" is anchored
  to the list's *own* publish date, not `T` (so list entries are often already
  expired by `T`),
- **social posts** that are undated or relative-only, sometimes with a
  thread-reply correction that flips an item's state ("this is expired now"),
- **cross-source conflicts** (same item, different dates/states in two docs).

Ground truth is **code-owned and deterministic**: every item's state at `T`
(ACTIVE / EXPIRED / FUTURE / UNDATED_AMBIGUOUS) is computed from its dates by
`generator.compute_state`, never asserted by an LLM. Rewards are
model-independent and reproducible.

## Architecture

```
generator.py   deterministic temporal DocBundle generator (seeded). Item state
               computed from dates; real brand names/domains; aggregator + social
               source types with traps; relative/explicit/vague/missing date
               labels; cross-source conflicts; structural-variety audit.
env.py         HUD v6 Environment. Hosts FastMCP tools — search, fetch,
               get_current_time, submit_answer — over a pre-built per-task
               corpus. 4 @env.template tasks (deal / event / status / summary).
grader.py      standalone programmatic grader. Scores the structured
               submit_answer payload (items + supporting_doc_ids) against the
               bundle's code-computed ground truth. No LLM judge, no free-text
               parsing -> rewards don't depend on the base model's prose habits.
tasks.py       builds the 140-task eval/train Taskset from queries.yaml.
train.py       on-policy RL loop (HUD TrainingClient + cispo loss, GRPO via
               group_size) over a forked trainable Qwen3 8B (Tinker).
queries.yaml   query templates expanded by the generator.
Dockerfile.hud / pyproject.toml / uv.lock — hackathon submission packaging.
```

## Scoring (model-independent)

The agent calls `submit_answer(items, supporting_doc_ids)`. An item counts only
if it is **grounded** — the agent cited at least one corpus doc that actually
contains that item. This kills the "bail to memorized brand names" failure mode
(reciting ChatGPT Plus / Claude Pro from memory without citing a fetched page
scores ~0). Free text in the agent's final message is never graded.

```
recall    = grounded gold items found / |gold|
precision = grounded gold / (grounded gold + grounded lures)
reward    = 0.5 * recall + 0.5 * precision     # 0 flat if submit_answer never called
```

A crafted perfect answer scores 1.0; a lure-asserted-as-current scores 0.0; a
no-citation bail scores 0.0; no `submit_answer` scores 0.0.

## Run

Local smoke (no API key — uses the deterministic `--no-prose` generator prose):

```bash
uv sync
python env.py                 # exercises tools + grader directly
hud serve env.py              # serve the v6 control channel
hud eval tasks.py <agent> --group 4 --config temperature=1.0
```

Generator checks + structural-variety audit:

```bash
python generator.py --seed 42 check     # determinism + sample dump
python generator.py audit               # 500-seed fingerprint audit
```

On-policy RL fine-tuning (needs `HUD_API_KEY`; fork a trainable base once):

```bash
hud models fork 22b93b24-e8e6-4864-8083-0a2a8b987c88 --name tempora-qwen3-8b
# 22b93b24... = "Qwen3 8B (Tinker)"
python train.py --steps 30 --tasks-per-step 4 --group 4 \
    --max-steps 20 --max-concurrent 8 --lr 2e-6 --temperature 1.0
```

`train.py` auto-loads `~/.hud/.env`. Per-step reward + per-task rewards +
checkpoint head are appended to `train_log.jsonl`.

## Why this is a real RL problem

Same task, same corpus, temperature>0 sampling produces high reward spread
(e.g. deal task 0.43 ± 0.43 — half the rollouts engage + cite + exclude lures,
half bail to memorized knowledge and score 0). That is exactly the
distribution GRPO learns from: the gradient pushes toward grounding answers in
*fetched* sources and excluding temporally-invalid items, away from
memorized-knowledge bailouts. The structured grader makes the signal
independent of which base model is trained, so the learned behavior transfers.

## Results

We fine-tuned **GPT-OSS-20B** on-policy with **GRPO** (HUD `TrainingClient`,
CISPO loss). We chose GPT-OSS-20B because it reliably calls `submit_answer`
(4/4 in probes) where dense-8B / 3B-active-MoE bases mostly bail without
submitting — a cold-start trap for GRPO (all-zero groups give no gradient).

Measured on a **held-fixed eval slice** (same tasks before vs. after training):

| Slice | Base reward | Trained | Δ | Bail rate (base → trained) |
|-------|------------:|--------:|----:|---------------------------|
| Mixed (easy + hard) | 0.358 | 0.399 | +0.04 | 45% → 42% |
| **Hard (difficulty 0.5)** | **0.160** | **0.290** | **+0.13 (+81%)** | **75% → 57%** |

The gains concentrate exactly where temporal reasoning is hard:
**hard-task mean reward 0.16 → 0.29 (+81% relative)** with the no-answer "bail"
rate cut from **75% → 57%**. The effect is larger on hard tasks than mixed
because the base already solves the easy ones (little headroom) — so a hard-only
training run lifts the hard slice most, while still nudging the mixed average up
(0.358 → 0.399, no forgetting on easy tasks). Both numbers are from held-fixed
20-task slices — the same tasks before and after; per-step training means are
noisier.

## Design notes

- **Bundle timing:** one bundle per task, generated before the agent acts;
  `search` ranks that fixed corpus (it never generates new docs on the fly), so
  reward spread across a GRPO group is attributable to the model, not divergent
  docs.
- **Variety / anti-overfit:** the *shape* is sampled per bundle (doc/item
  counts, date-label mix, lure composition, gold/lure ratio, conflicts,
  search-placement adversity), not fixed. The variety audit reports distinct
  fingerprints + entropy over 500 seeds (target: many distinct, no dominant
  fingerprint).
- **Per-rollout state:** each task run brings up its own env process, so the
  FastMCP tools read a module-level `STATE` populated by the template before
  yielding the prompt.