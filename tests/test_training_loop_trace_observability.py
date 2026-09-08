"""Regression tests for training/loop.py's inequity-trace observability
wiring (Sec. 3.2): the trace vector passed to act() for a given step, and
stored in that step's rollout entry, must be causally one step behind --
computed from rewards up to and including the *previous* step, never the
step whose action is currently being chosen. A fake env/agent pair with
hand-computable rewards makes the exact sequence directly inspectable,
independent of any real network math (same pattern as
test_training_loop_rollout_alignment.py).
"""

import numpy as np
import pytest

from hughes2018.reward.inequity_aversion import InequityAversionReward
from hughes2018.training.loop import train


class _ScriptedRewardEnv:
    """Single agent, reward equal to the step index (1-indexed) so the
    trace's evolution is exactly hand-computable."""

    def __init__(self, episode_length=100):
        self.episode_length = episode_length
        self._t = 0

    def reset(self):
        self._t = 0
        return {"agent-0": np.array([self._t])}

    def step(self, actions):
        self._t += 1
        obs = {"agent-0": np.array([self._t])}
        reward = float(self._t)  # step 1 -> reward 1.0, step 2 -> reward 2.0, ...
        done = self._t >= self.episode_length
        return obs, {"agent-0": reward}, {"agent-0": done, "__all__": done}, {"agent-0": {}}


class _RecordingAgent:
    def __init__(self):
        self.lstm_state = None
        self.seen_traces = []
        self.last_rollout = None

    def reset_lstm_state(self):
        pass

    def act(self, obs, trace=None):
        self.seen_traces.append(list(trace) if trace is not None else None)
        return 0, 0.0, 0.0, None

    def value_only(self, obs, trace=None):
        return 0.0

    def update(self, rollout):
        self.last_rollout = rollout
        return {"loss": 0.0}


def test_trace_passed_to_act_is_causally_one_step_behind():
    env = _ScriptedRewardEnv()
    agent = _RecordingAgent()
    inequity_reward = InequityAversionReward(
        agent_ids=["agent-0"], alpha={"agent-0": 1.0}, beta={"agent-0": 1.0}, gamma=1.0, trace_lambda=0.9
    )
    train(env, {"agent-0": agent}, num_env_steps=4, rollout_length=4, inequity_reward=inequity_reward)

    # e^{-1}=0 (reset); e^0 = 0.9*0 + r_1=1 = 1.0; e^1 = 0.9*1.0 + r_2=2 = 2.9;
    # e^2 = 0.9*2.9 + r_3=3 = 5.61. act() for steps 1..4 should see, in
    # order: e^{-1}, e^0, e^1, e^2 -- never the trace already reflecting
    # that same step's own reward.
    expected = [[0.0], [1.0], [2.9], [5.61]]
    actual = [t[0] for t in agent.seen_traces]
    assert actual == pytest.approx([e[0] for e in expected])


def test_rollout_traces_match_what_act_was_actually_called_with():
    env = _ScriptedRewardEnv()
    agent = _RecordingAgent()
    inequity_reward = InequityAversionReward(
        agent_ids=["agent-0"], alpha={"agent-0": 1.0}, beta={"agent-0": 1.0}, gamma=1.0, trace_lambda=0.9
    )
    train(env, {"agent-0": agent}, num_env_steps=4, rollout_length=4, inequity_reward=inequity_reward)

    stored_traces = [list(t) for t in agent.last_rollout.traces]
    assert stored_traces == agent.seen_traces


def test_zero_filled_trace_when_no_inequity_reward_configured():
    env = _ScriptedRewardEnv()
    agent = _RecordingAgent()
    train(env, {"agent-0": agent}, num_env_steps=3, rollout_length=3, inequity_reward=None)
    assert agent.seen_traces == [[0.0]] * 3
    assert [list(t) for t in agent.last_rollout.traces] == [[0.0]] * 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
