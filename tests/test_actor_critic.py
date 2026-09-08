"""Smoke + correctness tests for the actor-critic agent."""

import numpy as np
import pytest
import torch

from hughes2018.agents.actor_critic import ActorCriticAgent, ActorCriticConfig, Rollout

OBS_H = OBS_W = 15


def _make_agent(**config_kwargs):
    return ActorCriticAgent(
        obs_channels=3, obs_height=OBS_H, obs_width=OBS_W, config=ActorCriticConfig(num_actions=9, **config_kwargs)
    )


def _random_obs():
    return (np.random.default_rng(0).integers(0, 255, size=(3, OBS_H, OBS_W))).astype(np.uint8)


def test_act_returns_valid_action_and_shapes():
    agent = _make_agent()
    action, log_prob, value, state_before = agent.act(_random_obs())
    assert 0 <= action < 9
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert state_before[0].shape == (1, agent.cfg.lstm_hidden)


def test_lstm_state_persists_across_act_calls_until_reset():
    agent = _make_agent(seed=0)
    initial = agent.lstm_state
    agent.act(_random_obs())
    after_one_step = agent.lstm_state
    assert not torch.equal(initial[0], after_one_step[0])  # state actually advanced
    agent.reset_lstm_state()
    assert torch.equal(agent.lstm_state[0], initial[0])


def test_update_runs_and_changes_every_named_parameter():
    # Regression test for a bug caught in review: a prior lazy-shape-
    # inference version of fc1 was silently excluded from the optimizer, so
    # `any(...)` here would pass even with fc1 completely frozen. Checking
    # *every* named parameter individually (not just "any changed") is what
    # actually catches that class of bug.
    torch.manual_seed(0)
    agent = _make_agent(learning_rate=1e-2)
    rollout = Rollout(initial_lstm_state=agent.network.initial_state(1, agent.device))
    for t in range(5):
        obs = _random_obs()
        action, _log_prob, _value, _state_before = agent.act(obs)
        rollout.obs.append(obs)
        rollout.actions.append(action)
        rollout.rewards.append(1.0 if t == 4 else 0.0)
        rollout.dones.append(False)

    before = {name: p.clone() for name, p in agent.network.named_parameters()}
    stats = agent.update(rollout)

    assert "loss" in stats and np.isfinite(stats["loss"])
    unchanged = [
        name for name, p in agent.network.named_parameters() if torch.equal(before[name], p)
    ]
    assert not unchanged, f"parameters that did not update: {unchanged}"


def test_save_load_roundtrip_reproduces_policy(tmp_path):
    agent = _make_agent(seed=0)
    # Train briefly so weights differ from their random init -- otherwise a
    # save/load bug that silently kept the freshly-initialized weights
    # instead of the saved ones could go unnoticed.
    rollout = Rollout(initial_lstm_state=agent.network.initial_state(1, agent.device))
    for t in range(5):
        obs = _random_obs()
        action, _lp, _v, _s = agent.act(obs)
        rollout.obs.append(obs)
        rollout.actions.append(action)
        rollout.rewards.append(1.0 if t == 4 else 0.0)
        rollout.dones.append(False)
    agent.update(rollout)

    checkpoint_path = tmp_path / "agent.pt"
    agent.save(checkpoint_path)

    fresh_agent = _make_agent(seed=999)  # different seed -> different random init
    before_load = {name: p.clone() for name, p in fresh_agent.network.named_parameters()}
    fresh_agent.load(checkpoint_path)

    for name, p in fresh_agent.network.named_parameters():
        assert not torch.equal(before_load[name], p), f"{name} unchanged by load()"
        assert torch.equal(dict(agent.network.named_parameters())[name], p), f"{name} doesn't match saved agent"


def _make_rollout(agent, num_steps=5, reward_on_last=1.0):
    rollout = Rollout(initial_lstm_state=agent.network.initial_state(1, agent.device))
    for t in range(num_steps):
        obs = _random_obs()
        action, _lp, _v, _s = agent.act(obs)
        rollout.obs.append(obs)
        rollout.actions.append(action)
        rollout.rewards.append(reward_on_last if t == num_steps - 1 else 0.0)
        rollout.dones.append(False)
    return rollout


def test_compute_then_apply_gradients_matches_update():
    # update() is composed from compute_gradients() + apply_gradients(); this
    # checks that composition actually produces the same result as calling
    # update() directly, from the same starting weights and rollout -- the
    # equivalence the parallel training path (which calls them separately,
    # across process boundaries) depends on.
    torch.manual_seed(0)
    agent_a = _make_agent(seed=1, learning_rate=1e-2)
    torch.manual_seed(0)
    agent_b = _make_agent(seed=1, learning_rate=1e-2)
    for name, p in agent_a.network.named_parameters():
        assert torch.equal(p, dict(agent_b.network.named_parameters())[name])  # same starting weights

    rollout_a = _make_rollout(agent_a)
    rollout_b = Rollout(
        initial_lstm_state=rollout_a.initial_lstm_state,
        obs=list(rollout_a.obs),
        actions=list(rollout_a.actions),
        rewards=list(rollout_a.rewards),
        dones=list(rollout_a.dones),
        bootstrap_value=rollout_a.bootstrap_value,
    )

    stats_direct = agent_a.update(rollout_a)
    grads, stats_split = agent_b.compute_gradients(rollout_b)
    agent_b.apply_gradients(grads)

    assert stats_direct == pytest.approx(stats_split)
    for name, p_a in agent_a.network.named_parameters():
        p_b = dict(agent_b.network.named_parameters())[name]
        assert torch.allclose(p_a, p_b, atol=1e-6), f"{name} diverged between update() and compute+apply"


def test_apply_gradients_averaged_across_two_grad_sets_differs_from_either_alone():
    # Sanity check for the parallel training path's core operation: applying
    # the average of two different gradient sets should move weights
    # differently than applying either set alone.
    agent = _make_agent(seed=0, learning_rate=1e-1)
    rollout_1 = _make_rollout(agent, reward_on_last=1.0)
    grads_1, _ = agent.compute_gradients(rollout_1)
    agent.lstm_state = agent.network.initial_state(1, agent.device)  # don't let act() calls above bleed in
    rollout_2 = _make_rollout(agent, reward_on_last=-1.0)
    grads_2, _ = agent.compute_gradients(rollout_2)

    averaged = {name: (grads_1[name] + grads_2[name]) / 2.0 for name in grads_1}
    before = {name: p.clone() for name, p in agent.network.named_parameters()}
    agent.apply_gradients(averaged)
    after = dict(agent.network.named_parameters())

    changed = any(not torch.equal(before[name], after[name]) for name in before)
    assert changed
    # The averaged update should differ from what either gradient set alone
    # would have produced (they push in different directions given the
    # opposite-sign rewards).
    assert not all(torch.equal(after[name], before[name] - 1e-1 * grads_1[name]) for name in before)


def test_trace_dim_zero_by_default_act_ignores_a_passed_trace():
    # trace_dim defaults to 0 -- passing a trace anyway must be a silent
    # no-op (the network never concatenates it in), not an error, so
    # existing callers that don't know about this feature stay unaffected.
    # Compares value_only()'s deterministic output rather than act()'s
    # sampled action: two act() calls on identical logits can still sample
    # different actions (torch's global RNG advances between calls), so
    # only the value head -- unaffected by that sampling -- is a valid
    # equality check here.
    agent = _make_agent(seed=0)
    value_without_trace = agent.value_only(_random_obs())
    value_with_ignored_trace = agent.value_only(_random_obs(), trace=[1.0, 2.0, 3.0])
    assert value_without_trace == pytest.approx(value_with_ignored_trace)


def test_trace_dim_positive_requires_a_trace_argument():
    agent = _make_agent(trace_dim=3)
    with pytest.raises(ValueError):
        agent.act(_random_obs())  # no trace passed, but trace_dim=3


def test_trace_dim_positive_accepts_a_trace_and_changes_output():
    # A different trace vector for the same obs/LSTM-state should reach the
    # network (via the concatenated fc1 input) and change its output --
    # otherwise the trace input would be silently disconnected.
    agent = _make_agent(seed=0, trace_dim=3)
    obs = _random_obs()
    _action, _log_prob, value_zero, _ = agent.act(obs, trace=[0.0, 0.0, 0.0])
    agent.reset_lstm_state()
    _action, _log_prob, value_nonzero, _ = agent.act(obs, trace=[10.0, -5.0, 3.0])
    assert value_zero != pytest.approx(value_nonzero)


def test_update_with_trace_dim_changes_every_named_parameter_including_fc1():
    # Same rigor as test_update_runs_and_changes_every_named_parameter, but
    # with trace_dim>0: fc1's input width now includes the trace slice, and
    # this checks that slice's weights actually receive gradient too, not
    # just the image-derived slice.
    torch.manual_seed(0)
    agent = _make_agent(learning_rate=1e-2, trace_dim=3)
    rollout = Rollout(initial_lstm_state=agent.network.initial_state(1, agent.device))
    rng = np.random.default_rng(1)
    for t in range(5):
        obs = _random_obs()
        trace = list(rng.uniform(-1, 1, size=3))
        action, _log_prob, _value, _state_before = agent.act(obs, trace)
        rollout.obs.append(obs)
        rollout.actions.append(action)
        rollout.rewards.append(1.0 if t == 4 else 0.0)
        rollout.dones.append(False)
        rollout.traces.append(trace)

    before = {name: p.clone() for name, p in agent.network.named_parameters()}
    agent.update(rollout)
    unchanged = [name for name, p in agent.network.named_parameters() if torch.equal(before[name], p)]
    assert not unchanged, f"parameters that did not update: {unchanged}"


def test_update_return_matches_hand_computed_discounted_sum():
    agent = _make_agent(discount=0.9)
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
