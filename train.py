"""On-policy RL fine-tuning for the Tempora environment (HUD v6 cookbook).

Runs GRPO-style rollouts of a trainable (Tinker-backed) model against ``env.py``
via ``Taskset.run``, then feeds the resulting ``Run`` objects (prompt + output
tokens + logprobs + reward) straight to ``TrainingClient.step`` with the
``cispo`` loss (HUD's clipped-importance-sampling policy optimization — the
GRPO-style loss; advantages are normalized within each contiguous group of
``group_size``).

Per step:
  1. sample K tasks from the 140-task taskset (variety each step),
  2. run each task `group` times (same task, `group` rollouts -> reward spread
     -> the GRPO gradient), agent sampling at temperature>0 so rollouts differ,
  3. order runs per-task (contiguous) so group_size groups each task together,
  4. one client.step (forward+backward+optim) on the batch,
  5. log mean reward + per-task rewards + the new checkpoint head.

The agent is the SAME forked model the TrainingClient controls, so rollouts use
the current policy weights and each optim step moves the policy the rollouts
were sampled from (on-policy).

Prereq: fork a trainable base once, e.g.::

    hud models fork 22b93b24-e8e6-4864-8083-0a2a8b987c88 --name tempora-qwen3-8b
    # (22b93b24... = "Qwen3 8B (Tinker)") -> use the returned slug/id as MODEL.

Run::

    python train.py --steps 20 --tasks-per-step 8 --group 4 \\
        --lr 1e-6 --max-steps 20 --temperature 1.0

HUD_API_KEY must be set (the gateway routes the forked model + records tokens).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path


def _load_dotenv() -> None:
    """Populate os.environ from ~/.hud/.env and ./.env (only keys not already set).

    The `hud` CLI auto-loads ~/.hud/.env; a bare `python train.py` does not, so we
    load it ourselves so rollouts route through the gateway with the forked model.
    """
    for p in (Path.home() / ".hud" / ".env", Path.cwd() / ".env"):
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()

from hud import Taskset
from hud.agents import create_agent
from hud.eval import LocalRuntime
from hud.settings import settings
from hud.train import TrainingClient
from openai import AsyncOpenAI

from tasks import TASKSET


def _gateway_client(max_retries: int, timeout: float) -> AsyncOpenAI:
    """An OpenAI client pointed at the HUD gateway (same as
    ``gateway.build_gateway_client('openai')``) but with SDK-level retry on
    429/5xx/timeout. Tinker (the trainable backend) intermittently 503s with
    ``no healthy keys / upstream_overloaded`` under hackathon load; without
    retry, one 503 mid-rollout ends the run with no submit_answer -> a 0 that
    pollutes the GRPO batch with infra noise. Retries salvage transient 503s.
    """
    return AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.hud_gateway_url,
        max_retries=max_retries,
        timeout=timeout,
    )

# Forked GPT-OSS-20B (Tinker) "tempora-gptoss20b". Chosen over Qwen3 8B because
# the 8B has weak tool-calling discipline — it frequently ends a turn with a
# plain-text answer instead of calling submit_answer, so the grader scores 0 and
# a whole GRPO group can be all-zeros (no gradient: a cold-start trap). GPT-OSS-20B
# reliably calls submit_answer (4/4 in probes) with real reward spread. Override
# with TEMPORA_MODEL.
# Address the fork by its SLUG (the "gateway model id"), NOT the UUID. Inference
# by UUID returned 400 "model not supported" after a ~3am gateway change; the slug
# is the stable id you sample/train by. Training endpoints accept the UUID too, but
# the slug works for both, so use it everywhere.
DEFAULT_MODEL = "tempora-gptoss20b"
ENV_PATH = "env.py"
LOG_PATH = Path("train_log.jsonl")


def _sample_slugs(all_slugs: list[str], k: int, rng: random.Random) -> list[str]:
    return rng.sample(all_slugs, min(k, len(all_slugs)))


def _usable(r) -> bool:
    """A run we can train on: it has a real trajectory (trace_id) and didn't
    error. Run.failed is a *builder* method (`failed(error)->Run`), not a status
    flag, so don't use it. A pre-launch failure has ``trace_id=None`` (reward
    defaults to 0.0, indistinguishable from a real 0-reward rollout by reward
    alone); a mid-run failure keeps its trace but sets ``trace.status='error'``.
    """
    if r.trace_id is None:
        return False
    if getattr(getattr(r, "trace", None), "status", None) == "error":
        return False
    return True


def _ordered_runs(job, slugs: list[str], group: int) -> list:
    """Runs ordered per-task (contiguous), one full group per task.

    TrainingClient normalizes advantages within contiguous groups of
    `group_size`, so each task's `group` rollouts must be adjacent. Tasks with
    fewer than `group` usable rollouts are dropped (a partial group has no
    within-group baseline). Extras beyond `group` are trimmed.
    """
    runs = []
    for slug in slugs:
        rs = [r for r in job.results.get(slug, []) if _usable(r)]
        if len(rs) < group:
            continue  # can't form a full group -> no usable baseline
        runs.extend(rs[:group])
    return runs


async def train(args) -> None:
    model = args.model or os.environ.get("TEMPORA_MODEL", DEFAULT_MODEL)
    client = TrainingClient(model)
    # extra_body.return_token_ids is the flag that makes the openai-compatible
    # agent request token ids + logprobs from the Tinker backend and record them
    # on each AgentStep.sample — without it the gateway returns text only and
    # TrainingClient rejects the batch with "no trainable turns in the inputs".
    # model_client: a gateway AsyncOpenAI with SDK retry so transient Tinker 503s
    # (upstream_overloaded) are retried instead of killing the rollout.
    agent = create_agent(
        model,
        max_steps=args.max_steps,
        completion_kwargs={
            "temperature": args.temperature,
            "extra_body": {"return_token_ids": True},
        },
        model_client=_gateway_client(max_retries=args.max_retries, timeout=args.request_timeout),
    )
    runtime = LocalRuntime(ENV_PATH)

    all_slugs = list(TASKSET.tasks.keys())
    if args.min_difficulty is not None:
        # Concentrate training signal on hard tasks: the easy ones the base
        # already solves (no headroom -> near-zero gradient), so they only
        # dilute the batch. Filter to tasks at/above the difficulty threshold.
        all_slugs = [s for s in all_slugs
                     if (getattr(TASKSET.tasks[s], "args", {}) or {}).get("difficulty", 0) >= args.min_difficulty]
        print(f"difficulty filter >= {args.min_difficulty}: {len(all_slugs)} tasks")
    rng = random.Random(args.seed)
    head = await client.head()
    print(f"model={model}  loss={args.loss}  group={args.group}  "
          f"tasks/step={args.tasks_per_step}  lr={args.lr}  temp={args.temperature}")
    print(f"starting head: {head}")
    print(f"taskset: {len(all_slugs)} tasks  | env={ENV_PATH}")
    print("-" * 78)

    # Overnight-robust loop: count only COMPLETED optim steps toward the target.
    # A step that yields too few usable runs (backend down / network drop) or
    # raises is RETRIED after backoff rather than burned — so a transient outage
    # pauses progress instead of silently consuming the remaining budget. We give
    # up only after `max_consecutive_failures` in a row (a sustained outage).
    step_i = 0
    consecutive_failures = 0
    while step_i < args.steps:
        slugs = _sample_slugs(all_slugs, args.tasks_per_step, rng)
        sub = Taskset("tempora-train", [TASKSET.tasks[s] for s in slugs])
        t0 = time.time()
        try:
            job = await sub.run(
                agent,
                runtime=runtime,
                group=args.group,
                max_concurrent=args.max_concurrent,
                rollout_timeout=args.rollout_timeout,
            )
            runs = _ordered_runs(job, slugs, args.group)
            if len(runs) < args.group:
                raise RuntimeError(f"too few usable runs ({len(runs)}/{args.group}) — backend likely degraded")

            rewards = [r.reward for r in runs]
            mean_r = sum(rewards) / len(rewards)
            var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards) if rewards else 0.0
            n_tasks_used = len(runs) // args.group

            res = await client.step(
                runs,
                learning_rate=args.lr,
                loss_fn=args.loss,
                group_size=args.group,
                reward_scale=args.reward_scale,
                num_substeps=args.num_substeps,
                weight_decay=args.weight_decay,
            )
            head = await client.head()
        except Exception as exc:  # noqa: BLE001 — overnight resilience: never die on a transient
            consecutive_failures += 1
            backoff = min(300, 15 * (2 ** (consecutive_failures - 1)))  # 15s,30s,60s,...cap 5min
            print(f"[step {step_i}] FAILED ({consecutive_failures}/{args.max_consecutive_failures}): "
                  f"{type(exc).__name__}: {str(exc)[:140]} — retrying same step in {backoff}s")
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"aborting: {consecutive_failures} consecutive failures — backend down too long. "
                      f"completed {step_i} steps; safe to resume later from the cloud head.")
                break
            await asyncio.sleep(backoff)
            continue  # retry the SAME step_i (do not advance, do not burn budget)

        consecutive_failures = 0  # a clean step resets the failure streak
        dt = time.time() - t0
        # head carries the just-completed step's training stats (num_tokens,
        # mean_reward, metrics) — the server-side view of the same batch.
        hmetrics = getattr(head, "metrics", None) or {}
        h_ntok = getattr(head, "num_tokens", None)
        h_mean = getattr(head, "mean_reward", None)
        line = (f"[step {step_i:>3}] runs={len(runs)} tasks={n_tasks_used} "
                f"mean_reward={mean_r:.3f} ±{var ** 0.5:.3f}  "
                f"lr={args.lr}  head={_head_id(head)}  "
                f"tokens={h_ntok} srv_mean={h_mean}  {dt:.0f}s")
        print(line)

        per_task = {
            slug: [round(r.reward, 3) for r in job.results.get(slug, [])][: args.group]
            for slug in slugs
        }
        LOG_PATH.open("a").write(json.dumps({
            "step": step_i, "model": model, "loss": args.loss,
            "group": args.group, "tasks_per_step": args.tasks_per_step,
            "runs": len(runs), "mean_reward": mean_r,
            "reward_std": var ** 0.5, "rewards": rewards,
            "per_task": per_task, "lr": args.lr, "head": _head_id(head),
            "head_tokens": h_ntok, "head_mean_reward": h_mean,
            "head_metrics": hmetrics,
            "elapsed_s": round(dt, 1), "ts": time.time(),
        }) + "\n")
        step_i += 1  # advance only on a COMPLETED optim step

    print("-" * 78)
    print(f"done. {step_i} steps. final head: {_head_id(head)}")
    print(f"log: {LOG_PATH.resolve()}")


def _head_id(head) -> str:
    return getattr(head, "id", None) or str(head) if head else "(none)"


def main() -> None:
    p = argparse.ArgumentParser(description="Tempora on-policy RL training (HUD v6).")
    p.add_argument("--model", default=None, help="forked trainable model slug/id (default: tempora-qwen3-8b fork)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--tasks-per-step", type=int, default=8, help="K: tasks sampled per step")
    p.add_argument("--group", type=int, default=4, help="rollouts per task (GRPO group size)")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max-steps", type=int, default=20, help="agent tool-call steps per rollout")
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7, help="sampling temp (>0 for GRPO variance; 0.7 keeps tool-calling reliable)")
    p.add_argument("--rollout-timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=8, help="SDK retries on 429/5xx (Tinker 503s)")
    p.add_argument("--max-consecutive-failures", type=int, default=20,
                   help="abort after this many consecutive failed steps (sustained outage)")
    p.add_argument("--request-timeout", type=float, default=180.0, help="per LLM request timeout (s)")
    p.add_argument("--loss", default="cispo", choices=["cispo", "ppo", "importance_sampling", "dro", "cross_entropy"])
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--num-substeps", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-difficulty", type=float, default=None,
                   help="train only on tasks with difficulty >= this (e.g. 0.5 for hard-only)")
    args = p.parse_args()
    asyncio.run(train(args))


if __name__ == "__main__":
    main()