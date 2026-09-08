"""Regression test for a bug caught in review: `train()` was storing each
step's *post-step* observation (o_{t+1}) paired with the action actually
chosen from the *pre-step* observation (o_t), silently training
`log_prob(a_t | o_{t+1})` instead of `log_prob(a_t | o_t)` and shifting the
whole LSTM replay sequence by one step. A fake env/agent pair makes both
sequences directly inspectable, independent of any real network math.
"""

import numpy as np

from hughes2018.training.loop import train


class _CountingEnv:
    """obs is just the step counter, so "which observation" is trivially
    identifiable without needing a real gridworld."""

    def __init__(self, episode_length=5):
        self.episode_length = episode_length
        self._t = 0

    def reset(self):
        self._t = 0
        return {"agent-0": np.array([self._t])}

    def step(self, actions):
        self._t += 1
        obs = {"agent-0": np.array([self._t])}
        done = self._t >= self.episode_length
        return obs, {"agent-0": 0.0}, {"agent-0": done, "__all__": done}, {"agent-0": {}}


class _RecordingAgent:
    """Duck-types ActorCriticAgent's interface; records exactly what it was
    called with instead of doing any real network math."""

    def __init__(self):
        self.lstm_state = None
        self.seen_obs = []
        self.last_rollout = None

    def reset_lstm_state(self):
        pass

    def act(self, obs, trace=None):
        self.seen_obs.append(int(obs[0]))
        return 0, 0.0, 0.0, None

    def value_only(self, obs, trace=None):
        return 0.0

    def update(self, rollout):
        self.last_rollout = rollout
        return {"loss": 0.0}


def test_rollout_obs_matches_what_act_was_actually_called_with():
    env = _CountingEnv(episode_length=100)  # long enough to avoid an episode boundary mid-rollout
    agent = _RecordingAgent()
    train(env, {"agent-0": agent}, num_env_steps=5, rollout_length=5)

    assert agent.seen_obs == [0, 1, 2, 3, 4]
    stored_obs = [int(o[0]) for o in agent.last_rollout.obs]
    # The bug this guards against would make stored_obs == [1, 2, 3, 4, 5]
    # (the post-step observation) instead of matching seen_obs exactly.
    assert stored_obs == agent.seen_obs == [0, 1, 2, 3, 4]
