#!/usr/bin/env python3
# ruff: noqa: E402
"""Experiment 1: baseline (selfish) vs. inequity-averse agents on Cleanup.

Reproduces the paper's headline comparison (Hughes et al. 2018, Sec. 4):
does the inequity-aversion reward improve collective return in Cleanup?

Defaults to a small step count as a smoke test that exercises the full
pipeline (env -> independent actor-critic training -> inequity aversion ->
comparison), the same convention the sibling Leibo2017 repo's run_*.py
scripts use -- it does not claim converged, paper-scale results. Pass
--total-steps with a much larger value (and patience/GPU) to attempt that.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hughes2018.agents.actor_critic import ActorCriticAgent, ActorCriticConfig
from hughes2018.envs.cleanup import CleanupEnv
from hughes2018.envs.grid_engine import GridWorldConfig
from hughes2018.reward.inequity_aversion import InequityAversionReward
from hughes2018.training.loop import train


def run_condition(name: str, use_inequity_reward: bool, args) -> dict:
    print(f"\n=== Condition: {name} ===")
    rng = np.random.default_rng(args.seed)
    env = CleanupEnv(num_agents=args.num_agents, config=GridWorldConfig(episode_length=args.episode_length), rng=rng)
    obs_hw = 2 * env.cfg.view_radius + 1
    agents = {
        f"agent-{i}": ActorCriticAgent(
            obs_channels=3,
            obs_height=obs_hw,
            obs_width=obs_hw,
            config=ActorCriticConfig(num_actions=env.num_actions, seed=args.seed + i, learning_rate=args.lr),
            device=args.device,
        )
        for i in range(args.num_agents)
    }
    inequity_reward = None
    if use_inequity_reward:
        agent_ids = list(agents.keys())
        alpha_rng = np.random.default_rng(args.seed + 1000)
        # alpha ~ U(2.4, 3.0), beta ~ U(0.16, 0.20): the same population-
        # heterogeneity sampling range used for the reputation reward in the
        # sibling SequentialSocialDilemmas repo's cleanup_reputation
        # experiment (documented there as inherited from this lineage of
        # papers' typical parameterization; not a verified reproduction of a
        # specific number this paper states, since this paper's own
        # alpha/beta ranges aren't quoted here from primary text -- see
        # README "What's simplified vs. the paper").
        inequity_reward = InequityAversionReward(
            agent_ids=agent_ids,
            alpha={aid: float(alpha_rng.uniform(2.4, 3.0)) for aid in agent_ids},
            beta={aid: float(alpha_rng.uniform(0.16, 0.20)) for aid in agent_ids},
            trace_lambda=args.inequity_trace_lambda,
        )

    def on_log(stats):
        recent = stats.episode_returns[-5:]
        mean_recent = np.mean(recent) if recent else float("nan")
        print(f"  steps={stats.total_steps:>8} episodes={stats.total_episodes:>5} recent_return={mean_recent:.2f}")

    stats = train(
        env,
        agents,
        num_env_steps=args.total_steps,
        rollout_length=args.rollout_length,
        inequity_reward=inequity_reward,
        on_log=on_log,
    )
    return {
        "condition": name,
        "total_steps": stats.total_steps,
        "total_episodes": stats.total_episodes,
        "episode_returns": stats.episode_returns,
        "mean_collective_return": float(np.mean(stats.episode_returns)) if stats.episode_returns else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--total-steps", type=int, default=20_000, help="Smoke-test scale by default.")
    parser.add_argument("--rollout-length", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--inequity-trace-lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="output/run_experiment1_cleanup")
    args = parser.parse_args()

    results = {
        "baseline": run_condition("baseline (selfish)", use_inequity_reward=False, args=args),
        "inequity_averse": run_condition("inequity-averse", use_inequity_reward=True, args=args),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    baseline_mean = results["baseline"]["mean_collective_return"]
    inequity_mean = results["inequity_averse"]["mean_collective_return"]
    print("\n=== Summary ===")
    print(f"baseline mean collective return:        {baseline_mean}")
    print(f"inequity-averse mean collective return: {inequity_mean}")
    print(f"Wrote results to {out_dir / 'results.json'}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.bar(["baseline", "inequity-averse"], [baseline_mean or 0, inequity_mean or 0])
        ax.set_ylabel("Mean collective return")
        ax.set_title("Cleanup: baseline vs. inequity-averse agents")
        fig.savefig(out_dir / "collective_return.png")
        print(f"Wrote plot to {out_dir / 'collective_return.png'}")
    except ImportError:
        print("matplotlib not installed; skipping plot (results.json still written).")


if __name__ == "__main__":
    main()
