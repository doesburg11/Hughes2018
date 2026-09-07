"""End-to-end smoke test: env + independent actor-critic agents + inequity
aversion, wired together through the training loop, for a small number of
steps. This is the same role `tests/test_training_smoke.py` plays in the
sibling Leibo2017 repo -- exercises the full pipeline, not a claim of
converged or paper-scale results."""

import numpy as np
import pytest

from hughes2018.agents.actor_critic import ActorCriticAgent, ActorCriticConfig
from hughes2018.envs.cleanup import CleanupEnv
from hughes2018.envs.grid_engine import GridWorldConfig
from hughes2018.envs.harvest import HarvestEnv
from hughes2018.reward.inequity_aversion import InequityAversionReward
from hughes2018.training.loop import train


def _make_agents(num_agents: int, num_actions: int, obs_hw: int, seed: int = 0):
    agents = {}
    for i in range(num_agents):
        agent_id = f"agent-{i}"
        agents[agent_id] = ActorCriticAgent(
            obs_height=obs_hw,
            obs_width=obs_hw,
            obs_channels=3,
            config=ActorCriticConfig(num_actions=num_actions, seed=seed + i),
        )
    return agents


@pytest.mark.parametrize("use_inequity_reward", [False, True])
def test_cleanup_training_smoke(use_inequity_reward):
    num_agents = 2
    env = CleanupEnv(
        num_agents=num_agents,
        config=GridWorldConfig(height=12, width=16, episode_length=15),
        rng=np.random.default_rng(0),
    )
    agents = _make_agents(num_agents, env.num_actions, obs_hw=2 * env.cfg.view_radius + 1)
    inequity_reward = None
    if use_inequity_reward:
        agent_ids = list(agents.keys())
        inequity_reward = InequityAversionReward(
            agent_ids=agent_ids,
            alpha={aid: 2.7 for aid in agent_ids},
            beta={aid: 0.18 for aid in agent_ids},
        )
    stats = train(env, agents, num_env_steps=40, rollout_length=10, inequity_reward=inequity_reward)
    assert stats.total_steps >= 40
    assert len(stats.update_logs) > 0
    assert all(np.isfinite(log["loss"]) for log in stats.update_logs)


def test_harvest_training_smoke():
    num_agents = 2
    env = HarvestEnv(
        num_agents=num_agents,
        config=GridWorldConfig(height=10, width=14, episode_length=15),
        rng=np.random.default_rng(0),
    )
    agents = _make_agents(num_agents, env.num_actions, obs_hw=2 * env.cfg.view_radius + 1)
    stats = train(env, agents, num_env_steps=40, rollout_length=10)
    assert stats.total_steps >= 40
    assert all(np.isfinite(log["loss"]) for log in stats.update_logs)


def test_episode_boundary_resets_lstm_and_inequity_state():
    num_agents = 2
    env = CleanupEnv(
        num_agents=num_agents,
        config=GridWorldConfig(height=12, width=16, episode_length=5),  # short episodes force resets
        rng=np.random.default_rng(1),
    )
    agents = _make_agents(num_agents, env.num_actions, obs_hw=2 * env.cfg.view_radius + 1)
    agent_ids = list(agents.keys())
    inequity_reward = InequityAversionReward(
        agent_ids=agent_ids, alpha={aid: 2.7 for aid in agent_ids}, beta={aid: 0.18 for aid in agent_ids}
    )
    stats = train(env, agents, num_env_steps=30, rollout_length=10, inequity_reward=inequity_reward)
    assert stats.total_episodes >= 3  # 30 steps / 5-step episodes
    assert len(stats.episode_returns) == stats.total_episodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
