"""Smoke + correctness tests for the actor-critic agent."""

import numpy as np
import pytest
import torch

from hughes2018.agents.actor_critic import ActorCriticAgent, ActorCriticConfig, Rollout


def _random_obs():
    return (np.random.default_rng(0).integers(0, 255, size=(3, 15, 15))).astype(np.uint8)


def test_act_returns_valid_action_and_shapes():
    agent = ActorCriticAgent(obs_channels=3, config=ActorCriticConfig(num_actions=9))
    action, log_prob, value, state_before = agent.act(_random_obs())
    assert 0 <= action < 9
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert state_before[0].shape == (1, agent.cfg.lstm_hidden)


def test_lstm_state_persists_across_act_calls_until_reset():
    agent = ActorCriticAgent(obs_channels=3, config=ActorCriticConfig(num_actions=9, seed=0))
    initial = agent.lstm_state
    agent.act(_random_obs())
    after_one_step = agent.lstm_state
    assert not torch.equal(initial[0], after_one_step[0])  # state actually advanced
    agent.reset_lstm_state()
    assert torch.equal(agent.lstm_state[0], initial[0])


def test_update_runs_and_changes_parameters():
    torch.manual_seed(0)
    agent = ActorCriticAgent(obs_channels=3, config=ActorCriticConfig(num_actions=9, learning_rate=1e-2))
    rollout = Rollout(initial_lstm_state=agent.network.initial_state(1, agent.device))
    for t in range(5):
        obs = _random_obs()
        action, _log_prob, _value, _state_before = agent.act(obs)
        rollout.obs.append(obs)
        rollout.actions.append(action)
        rollout.rewards.append(1.0 if t == 4 else 0.0)
        rollout.dones.append(False)

    before = [p.clone() for p in agent.network.parameters()]
    stats = agent.update(rollout)
    after = list(agent.network.parameters())

    assert "loss" in stats and np.isfinite(stats["loss"])
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed


def test_update_return_matches_hand_computed_discounted_sum():
    agent = ActorCriticAgent(obs_channels=3, config=ActorCriticConfig(num_actions=9, discount=0.9))
    rollout = Rollout(initial_lstm_state=agent.network.initial_state(1, agent.device))
    rewards = [1.0, 0.0, 2.0]
    for t, r in enumerate(rewards):
        obs = _random_obs()
        action, _lp, _v, _s = agent.act(obs)
        rollout.obs.append(obs)
        rollout.actions.append(action)
        rollout.rewards.append(r)
        rollout.dones.append(False)
    rollout.bootstrap_value = 5.0

    # G_2 = r_2 + gamma * bootstrap = 2 + 0.9*5 = 6.5
    # G_1 = r_1 + gamma * G_2 = 0 + 0.9*6.5 = 5.85
    # G_0 = r_0 + gamma * G_1 = 1 + 0.9*5.85 = 6.265
    stats = agent.update(rollout)
    expected_mean_return = (6.265 + 5.85 + 6.5) / 3
    assert stats["mean_return"] == pytest.approx(expected_mean_return, rel=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
