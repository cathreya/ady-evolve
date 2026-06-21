"""Evaluate a model on a fixed slice of the Tempora taskset (no training).

For the hackathon result: run the SAME fixed task slice against (a) the base
Qwen3 8B and (b) the trained fork head, and compare mean reward + bail fraction.
Mirrors train.py's rollout path (same agent config, same runtime) but without
the optim step, so the comparison is apples-to-apples with the training
distribution.

The task slice is sampled deterministically from a seed and printed, so the
same slice can be re-run across models. Set ``--which`` to a previously printed
comma-separated slug list to eval an exact slice.

Examples::

    # base (public Qwen3 8B)
    python eval.py --model "Qwen/Qwen3-8B" --n-tasks 20 --seed 7 --tag base
    # trained fork head
    python eval.py --model tempora-qwen3-8b --n-tasks 20 --seed 7 --tag trained

``--group``>1 with ``--temperature``>0 gives a reward distribution (matches
training); ``--group 1 --temperature 0`` gives one greedy rollout per task.
``return_token_ids`` is NOT set (eval doesn't train).
"""
import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path


def _load_dotenv() -> None:
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

from tasks import TASKSET

ENV_PATH = "env.py"
LOG_PATH = Path("eval_log.jsonl")


def _usable(r) -> bool:
    if r.trace_id is None:
        return False
    if getattr(getattr(r, "trace", None), "status", None) == "error":
        return False
    return True


async def evaluate(args) -> None:
    agent = create_agent(
        args.model,
        max_steps=args.max_steps,
        completion_kwargs={"temperature": args.temperature},
    )
    runtime = LocalRuntime(ENV_PATH)

    all_slugs = list(TASKSET.tasks.keys())
    if args.which:
        slugs = [s.strip() for s in args.which.split(",") if s.strip()]
    else:
        rng = random.Random(args.seed)
        slugs = rng.sample(all_slugs, min(args.n_tasks, len(all_slugs)))

    print(f"model={args.model}  tag={args.tag}  group={args.group}  "
          f"temp={args.temperature}  max_steps={args.max_steps}")
    print(f"slice ({len(slugs)} tasks, seed={args.seed}): {','.join(slugs)}")
    print("-" * 78)

    sub = Taskset("tempora-eval", [TASKSET.tasks[s] for s in slugs])
    t0 = time.time()
    job = await sub.run(
        agent,
        runtime=runtime,
        group=args.group,
        max_concurrent=args.max_concurrent,
        rollout_timeout=args.rollout_timeout,
    )
    dt = time.time() - t0

    all_rewards = []
    per_task = {}
    n_bail = 0  # 0.0 reward rollouts (the bail-to-memorized / no-submit failure)
    n_failed = 0
    for slug in slugs:
        rs = job.results.get(slug, [])
        rew = [r.reward for r in rs if _usable(r)]
        n_failed += len(rs) - len(rew)
        per_task[slug] = [round(x, 3) for x in rew]
        all_rewards.extend(rew)
        n_bail += sum(1 for x in rew if x <= 1e-9)

    mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    var = sum((x - mean) ** 2 for x in all_rewards) / len(all_rewards) if all_rewards else 0.0
    nonzero = sum(1 for x in all_rewards if x > 1e-9)
    print(f"mean_reward={mean:.3f} ±{var ** 0.5:.3f}  "
          f"rollouts={len(all_rewards)}  nonzero={nonzero}  "
          f"bail(0.0)={n_bail} ({n_bail/len(all_rewards)*100:.0f}%)  "
          f"failed={n_failed}  {dt:.0f}s")
    print("per-task:")
    for slug in slugs:
        rew = per_task[slug]
        m = sum(rew) / len(rew) if rew else 0.0
        print(f"  {slug:24s} n={len(rew)} mean={m:.3f} {rew}")

    LOG_PATH.open("a").write(json.dumps({
        "tag": args.tag, "model": args.model, "seed": args.seed,
        "group": args.group, "temperature": args.temperature,
        "max_steps": args.max_steps, "slugs": slugs,
        "mean_reward": mean, "reward_std": var ** 0.5,
        "n_rollouts": len(all_rewards), "n_nonzero": nonzero,
        "n_bail": n_bail, "n_failed": n_failed,
        "per_task": per_task, "elapsed_s": round(dt, 1), "ts": time.time(),
    }) + "\n")
    print(f"logged to {LOG_PATH.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a model on a fixed Tempora task slice.")
    p.add_argument("--model", required=True, help="gateway model slug/id (e.g. 'Qwen/Qwen3-8B' or tempora-qwen3-8b)")
    p.add_argument("--tag", default="run", help="label for the eval_log.jsonl entry")
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--seed", type=int, default=7, help="seed for the task slice sampling")
    p.add_argument("--which", default=None, help="comma-separated slug list to eval exactly (overrides sampling)")
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rollout-timeout", type=float, default=360.0)
    args = p.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()