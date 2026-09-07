#!/usr/bin/env python3
"""Experiment 3: a heterogeneous population -- some agents inequity-averse,
some selfish, trained together in the same group on Cleanup.

This is a first cut at the paper's question of whether inequity aversion is
robust to exploitation by purely selfish co-players (Hughes et al. 2018,
Sec. 4's discussion of population heterogeneity), not a full reproduction of
it: the paper studies this across evolutionary population dynamics /
multiple training generations with selection pressure on which trait
survives; this script trains a single fixed-composition population for one
run and reports each subgroup's own mean per-capita return, which is the
first-order version of the same question ("does being inequity-averse cost
or benefit an agent when its co-players aren't") without the generational
selection dynamics on top. That's a documented gap here, not a hidden one --
see the README.

`InequityAversionReward.apply()` tracks smoothed rewards across the whole
group (needed so the comparison term is against the true group average,
including the selfish agents) but this script only *uses* the adjusted
reward for the agents flagged inequity-averse; selfish agents train on their
raw, unadjusted reward.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hughes2018.agents.actor_critic import ActorCriticAgent, ActorCriticConfig, Rollout
from hughes2018.envs.cleanup import CleanupEnv
from hughes2018.envs.grid_engine import GridWorldConfig
from hughes2018.reward.inequity_aversion import InequityAversionReward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--num-inequity-averse", type=int, default=2, help="The rest are selfish.")
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--total-steps", type=int, default=20_000, help="Smoke-test scale by default.")
    parser.add_argument("--rollout-length", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--inequity-smoothing", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="output/run_experiment3_heterogeneous")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = CleanupEnv(num_agents=args.num_agents, config=GridWorldConfig(episode_length=args.episode_length), rng=rng)
    agent_ids = [f"agent-{i}" for i in range(args.num_agents)]
    inequity_averse_ids = set(agent_ids[: args.num_inequity_averse])
    print(f"Inequity-averse agents: {sorted(inequity_averse_ids)}")
    print(f"Selfish agents:         {sorted(set(agent_ids) - inequity_averse_ids)}")

    agents = {
        aid: ActorCriticAgent(
            obs_channels=3,
            config=ActorCriticConfig(num_actions=env.num_actions, seed=args.seed + i, learning_rate=args.lr),
            device=args.device,
        )
        for i, aid in enumerate(agent_ids)
    }
    alpha_rng = np.random.default_rng(args.seed + 1000)
    inequity_reward = InequityAversionReward(
        agent_ids=agent_ids,
        alpha={aid: float(alpha_rng.uniform(2.4, 3.0)) for aid in agent_ids},
        beta={aid: float(alpha_rng.uniform(0.16, 0.20)) for aid in agent_ids},
        smoothing=args.inequity_smoothing,
    )

    subgroup_returns = {"inequity_averse": [], "selfish": []}
    per_agent_episode_return = {aid: 0.0 for aid in agent_ids}

    obs = env.reset()
    for agent in agents.values():
        agent.reset_lstm_state()
    inequity_reward.reset()
    total_steps = 0

    while total_steps < args.total_steps:
        rollouts = {aid: Rollout(initial_lstm_state=agents[aid].lstm_state) for aid in agent_ids}
        for _ in range(args.rollout_length):
            actions = {aid: agents[aid].act(obs[aid])[0] for aid in agent_ids}
            obs, raw_rewards, dones, _infos = env.step(actions)
            adjusted = inequity_reward.apply(raw_rewards)
            total_steps += 1
            for aid in agent_ids:
                per_agent_episode_return[aid] += raw_rewards[aid]
                used_reward = adjusted[aid] if aid in inequity_averse_ids else raw_rewards[aid]
                rollouts[aid].obs.append(obs[aid])
                rollouts[aid].actions.append(actions[aid])
                rollouts[aid].rewards.append(used_reward)
                rollouts[aid].dones.append(bool(dones["__all__"]))

            if dones["__all__"]:
                for aid in agent_ids:
                    key = "inequity_averse" if aid in inequity_averse_ids else "selfish"
                    subgroup_returns[key].append(per_agent_episode_return[aid])
                    per_agent_episode_return[aid] = 0.0
                obs = env.reset()
                for agent in agents.values():
                    agent.reset_lstm_state()
                inequity_reward.reset()
                break
            if total_steps >= args.total_steps:
                break

        for aid in agent_ids:
            rollout = rollouts[aid]
            if not rollout.obs:
                continue
            rollout.bootstrap_value = 0.0 if rollout.dones[-1] else agents[aid].value_only(obs[aid])
            agents[aid].update(rollout)

        if total_steps % (args.rollout_length * 20) < args.rollout_length:
            mean_ia = np.mean(subgroup_returns["inequity_averse"][-10:]) if subgroup_returns["inequity_averse"] else float("nan")
            mean_sf = np.mean(subgroup_returns["selfish"][-10:]) if subgroup_returns["selfish"] else float("nan")
            print(f"  steps={total_steps:>8} inequity_averse_mean={mean_ia:.2f} selfish_mean={mean_sf:.2f}")

    results = {
        "num_agents": args.num_agents,
        "num_inequity_averse": args.num_inequity_averse,
        "inequity_averse_agent_ids": sorted(inequity_averse_ids),
        "per_capita_returns": subgroup_returns,
        "mean_inequity_averse_return": float(np.mean(subgroup_returns["inequity_averse"])) if subgroup_returns["inequity_averse"] else None,
        "mean_selfish_return": float(np.mean(subgroup_returns["selfish"])) if subgroup_returns["selfish"] else None,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Summary ===")
    print(f"mean per-capita return, inequity-averse subgroup: {results['mean_inequity_averse_return']}")
    print(f"mean per-capita return, selfish subgroup:         {results['mean_selfish_return']}")
    print(f"Wrote results to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
