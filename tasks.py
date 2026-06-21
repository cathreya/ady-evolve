"""Build a Tempora evaluation taskset and (optionally) run it against an agent.

Two ways to run, both needing a HUD model agent (so a HUD_API_KEY):

  # 1) HUD CLI (auto-spawns env.py as the environment):
  hud eval tasks.py claude --all
  hud eval tasks.py claude-sonnet-4-6 --group 4 --max-concurrent 4

  # 2) Programmatic (this file as a script):
  python tasks.py claude
  python tasks.py claude-sonnet-4-6

Without a key you can still drive the env manually — see README.md
(`hud serve env.py` + `hud client run`), which is verified to work keyless.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

from env import deal_query, event_query, status_query, summary_query
from hud import Taskset

TEMPLATES = {
    "deal": deal_query,
    "event": event_query,
    "status": status_query,
    "summary": summary_query,
}

CONFIG_PATH = Path(__file__).parent / "queries.yaml"


def build_taskset(path: str | Path = CONFIG_PATH) -> Taskset:
    cfg = yaml.safe_load(Path(path).read_text())
    tasks = []
    for q in cfg["queries"]:
        tmpl = TEMPLATES[q["kind"]]
        for seed in cfg["seeds"]:
            for diff in cfg["difficulties"]:
                tasks.append(tmpl(query=q["text"], seed=seed, difficulty=diff))
    return Taskset("tempora-eval", tasks)


# Module-level taskset so `hud eval tasks.py <agent>` can load it.
# NOTE: expose only the Taskset, not a redundant `TASKS = list(...)` alias.
# hud's module scanner collects from every Taskset AND every list-of-Task it
# finds, so exposing both double-counts every task and trips "duplicate task
# slugs". A single Taskset is enough for every loader path.
TASKSET = build_taskset()


async def run(agent_name: str = "claude", group: int = 1, max_concurrent: int = 4) -> None:
    from hud.agents import create_agent
    from hud.eval import LocalRuntime

    agent = create_agent(agent_name)
    job = await TASKSET.run(
        agent, runtime=LocalRuntime("env.py"), group=group, max_concurrent=max_concurrent
    )
    print(f"mean reward: {job.reward:.3f}  (n={len(job.runs)})")
    for slug, runs in job.results.items():
        rewards = [r.reward for r in runs]
        print(f"  {slug:40s} {rewards}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "claude"
    asyncio.run(run(name))