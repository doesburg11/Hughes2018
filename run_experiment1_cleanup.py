#!/usr/bin/env python3
# ruff: noqa: E402
"""Experiment 1: baseline vs. inequity-averse agents on Cleanup.

Reproduces the paper's headline Cleanup comparison (Hughes et al. 2018,
Fig. 3): does inequity aversion improve collective return? Critically, the
paper's Fig. 3 tests *advantageous* inequity aversion (guilt, the beta term
alone, alpha=0) and *disadvantageous* inequity aversion (envy, alpha alone)
*separately* -- Fig. 3(A-C) shows advantageous inequity aversion helps
Cleanup; Fig. 3(D-F) explicitly states "disadvantageous inequity aversion
does not promote greater cooperation in the Cleanup game." Neither panel
tests the two combined. This script runs three conditions:

  - "baseline": no inequity term (vanilla A3C-equivalent).
  - "advantageous_only": beta term only (alpha=0) -- the condition the
    paper's own Fig. 3(A-C) shows actually helps Cleanup.
  - "both": alpha and beta together (this repo's original, broader
    condition, kept for reference -- not one of the paper's own two tested
    Cleanup conditions, so a null/mixed result here doesn't contradict the
    paper the way a null "advantageous_only" result would).

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
from hughes2018.reward.inequity_aversion import InequityAversionReward, scaled_alpha_beta_range
from hughes2018.training.loop import train


def build_inequity_reward(
    agent_ids: list[str], inequity_mode: str, trace_lambda: float, seed: int
) -> InequityAversionReward | None:
    """inequity_mode: "none" | "advantageous_only" | "both" -- see module docstring.

    Split out from run_condition() specifically so the alpha=0 wiring for
    "advantageous_only" (the paper's actual tested Cleanup condition) can be
    checked directly in a test, rather than only indirectly via a full
    training run's return statistics -- a silently-wrong alpha here (e.g.
    accidentally sampling it instead of zeroing it) wouldn't otherwise be
    caught by anything short of a training run failing to reproduce the
    paper's result, which is exactly the kind of quiet calibration bug this
    repo has already hit more than once.
    """
    if inequity_mode == "none":
        return None
    alpha_rng = np.random.default_rng(seed + 1000)
    # Amplification-corrected ranges (see scaled_alpha_beta_range's
    # docstring): the base 2.4-3.0/0.16-0.20 range was tuned for a bounded,
    # reward-scale quantity elsewhere; this trace is unnormalized and runs
    # ~1/(1-gamma*trace_lambda) larger, so the base range is divided by that
    # factor to keep the effective penalty on a comparable scale to the
    # extrinsic reward.
    alpha_range, beta_range = scaled_alpha_beta_range(gamma=0.99, trace_lambda=trace_lambda)
    if inequity_mode == "advantageous_only":
        alpha = {aid: 0.0 for aid in agent_ids}  # no envy term -- matches the paper's Fig. 3(A-C) condition
    elif inequity_mode == "both":
        alpha = {aid: float(alpha_rng.uniform(*alpha_range)) for aid in agent_ids}
    else:
        raise ValueError(f"Unknown inequity_mode: {inequity_mode!r}")
    return InequityAversionReward(
        agent_ids=agent_ids,
        alpha=alpha,
        beta={aid: float(alpha_rng.uniform(*beta_range)) for aid in agent_ids},
        trace_lambda=trace_lambda,
    )


def run_condition(name: str, inequity_mode: str, args) -> dict:
    """inequity_mode: "none" | "advantageous_only" | "both" -- see module docstring."""
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
    inequity_reward = build_inequity_reward(
        list(agents.keys()), inequity_mode, args.inequity_trace_lambda, args.seed
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

    condition_slug = name.split(" ")[0].replace("(", "").replace(")", "")  # "baseline" / "inequity-averse"
    checkpoint_dir = Path(args.out_dir) / "checkpoints" / condition_slug
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for agent_id, agent in agents.items():
        agent.save(checkpoint_dir / f"{agent_id}.pt")
    print(f"  Wrote checkpoints to {checkpoint_dir}")

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
        "baseline": run_condition("baseline", inequity_mode="none", args=args),
        "advantageous_only": run_condition("advantageous-only (guilt)", inequity_mode="advantageous_only", args=args),
        "both": run_condition("inequity-averse (both)", inequity_mode="both", args=args),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    means = {k: v["mean_collective_return"] for k, v in results.items()}
    print("\n=== Summary ===")
    print(f"baseline mean collective return:                    {means['baseline']}")
    print(f"advantageous-only (guilt) mean collective return:   {means['advantageous_only']}  <- paper's tested condition")
    print(f"inequity-averse (both) mean collective return:      {means['both']}")
    print(f"Wrote results to {out_dir / 'results.json'}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        labels = ["baseline", "advantageous-only\n(guilt)", "inequity-averse\n(both)"]
        ax.bar(labels, [means["baseline"] or 0, means["advantageous_only"] or 0, means["both"] or 0])
        ax.set_ylabel("Mean collective return")
        ax.set_title("Cleanup: baseline vs. inequity-averse conditions")
        fig.savefig(out_dir / "collective_return.png")
        print(f"Wrote plot to {out_dir / 'collective_return.png'}")
    except ImportError:
        print("matplotlib not installed; skipping plot (results.json still written).")


if __name__ == "__main__":
    main()
