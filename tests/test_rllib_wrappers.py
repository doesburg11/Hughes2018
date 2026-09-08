"""Tests for the optional Ray RLlib backend's env wrapper.

Most of these drive GridWorldRLlibEnv directly against a small, fully
deterministic fake inner env -- the wrapper's own contract (obs flattening,
terminated/truncated split, inequity-reward application, reset behavior) is
what's under test here, not CleanupEnv/HarvestEnv's own (non-deterministic)
dynamics. A couple of smoke tests at the bottom exercise the real
make_cleanup_rllib_env/make_harvest_rllib_env factories for basic shape/range
sanity only. None of this spins up a Ray cluster or runs actual training --
that end-to-end path is covered by manually running run_rllib_train.py at
small scale (too slow for a unit test).
"""

import numpy as np
import pytest

# ray[rllib] is an optional extra (requirements-rllib.txt), not part of the
# base install -- skip this whole module rather than fail collection when
# it's absent, so `pytest tests/ -q` still passes on a base install.
pytest.importorskip("ray.rllib")

from hughes2018.envs.grid_engine import GridWorldConfig  # noqa: E402
from hughes2018.envs.rllib_wrappers import (  # noqa: E402
    GridWorldRLlibEnv,
    make_cleanup_rllib_env,
    make_harvest_rllib_env,
)

OBS_HW = 5  # 2*view_radius+1 with view_radius=2


class _FakeCfg:
    view_radius = 2


class FakeInnerEnv:
    """Minimal stand-in for CleanupEnv/HarvestEnv's dict-keyed API."""

    num_actions = 4

    def __init__(self, num_agents=2, episode_length=3):
        self.cfg = _FakeCfg()
        self.agents = {f"agent-{i}": object() for i in range(num_agents)}
        self.episode_length = episode_length
        self.rng = None
        self._t = 0
        self.reward_value = 1.0

    def reset(self):
        self._t = 0
        return {aid: np.zeros((3, OBS_HW, OBS_HW), dtype=np.uint8) for aid in self.agents}

    def step(self, action_dict):
        self._t += 1
        obs = {aid: np.full((3, OBS_HW, OBS_HW), 255, dtype=np.uint8) for aid in self.agents}
        rewards = {aid: self.reward_value for aid in self.agents}
        done = self._t >= self.episode_length
        dones = {"__all__": done}
        infos = {aid: {} for aid in self.agents}
        return obs, rewards, dones, infos


def _fake_env(**kwargs):
    return GridWorldRLlibEnv(lambda: FakeInnerEnv(**kwargs))


def test_reset_returns_flat_float_obs_in_unit_range_for_every_agent():
    env = _fake_env()
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)
    for aid in env.possible_agents:
        o = obs[aid]
        assert o.shape == (3 * OBS_HW * OBS_HW,)
        assert o.dtype == np.float32
        assert o.max() == 0.0  # FakeInnerEnv.reset() returns all-zero obs
        assert infos[aid] == {}


def test_step_converts_and_scales_obs_to_unit_range():
    env = _fake_env()
    env.reset(seed=0)
    obs, *_ = env.step({aid: 0 for aid in env.possible_agents})
    for aid in env.possible_agents:
        assert obs[aid].shape == (3 * OBS_HW * OBS_HW,)
        assert np.allclose(obs[aid], 1.0)  # FakeInnerEnv.step() returns all-255 obs -> /255.0


def test_step_reports_truncation_not_termination_at_episode_end():
    env = _fake_env(episode_length=3)
    env.reset(seed=0)
    action_dict = {aid: 0 for aid in env.possible_agents}
    for _ in range(2):
        _obs, _rew, terminations, truncations, _infos = env.step(action_dict)
        assert terminations["__all__"] is False
        assert truncations["__all__"] is False

    _obs, _rew, terminations, truncations, _infos = env.step(action_dict)
    assert terminations["__all__"] is False
    assert truncations["__all__"] is True
    assert all(terminations[aid] is False for aid in env.possible_agents)
    assert all(truncations[aid] is True for aid in env.possible_agents)


def test_no_inequity_kwargs_passes_raw_rewards_through():
    env = _fake_env()
    assert env._inequity_reward is None
    env.reset(seed=0)
    _obs, rewards, *_ = env.step({aid: 0 for aid in env.possible_agents})
    assert rewards == {aid: 1.0 for aid in env.possible_agents}


def test_inequity_kwargs_wires_a_reward_object_that_adjusts_rewards():
    agent_ids = ["agent-0", "agent-1", "agent-2"]
    inequity_kwargs = {
        "alpha": {aid: 3.0 for aid in agent_ids},
        "beta": {aid: 3.0 for aid in agent_ids},
        "trace_lambda": 0.95,
    }
    env = GridWorldRLlibEnv(lambda: FakeInnerEnv(num_agents=3), inequity_kwargs=inequity_kwargs)
    assert env._inequity_reward is not None
    assert env._inequity_reward.agent_ids == agent_ids

    env.reset(seed=0)
    # Force an imbalance directly on the trace so the penalty is guaranteed
    # non-zero and hand-computable, rather than depending on emergent
    # env dynamics or randomness.
    env._inequity_reward.trace["agent-0"] = 10.0
    _obs, rewards, *_ = env.step({aid: 0 for aid in agent_ids})

    # Reproduce InequityAversionReward.apply()'s math by hand for this step:
    # trace was {agent-0: 10.0, agent-1: 0.0, agent-2: 0.0} before this step;
    # decay = gamma*trace_lambda = 0.99*0.95 = 0.9405, reward_value = 1.0.
    decay = 0.99 * 0.95
    trace_before = {"agent-0": 10.0, "agent-1": 0.0, "agent-2": 0.0}
    trace_after = {aid: decay * trace_before[aid] + 1.0 for aid in agent_ids}
    expected = {}
    n = len(agent_ids)
    for aid in agent_ids:
        own = trace_after[aid]
        disadvantageous = sum(max(trace_after[o] - own, 0.0) for o in agent_ids if o != aid)
        advantageous = sum(max(own - trace_after[o], 0.0) for o in agent_ids if o != aid)
        penalty = (3.0 / (n - 1)) * disadvantageous + (3.0 / (n - 1)) * advantageous
        expected[aid] = 1.0 - penalty

    for aid in agent_ids:
        assert rewards[aid] == pytest.approx(expected[aid])
    # agent-0 started far ahead -> its "guilt" (beta) penalty should be the
    # largest, leaving it worse off than the other two.
    assert rewards["agent-0"] < rewards["agent-1"] == pytest.approx(rewards["agent-2"])


def test_reset_reseeds_underlying_rng_and_resets_inequity_trace():
    inequity_kwargs = {
        "alpha": {"agent-0": 1.0, "agent-1": 1.0},
        "beta": {"agent-0": 1.0, "agent-1": 1.0},
        "trace_lambda": 0.95,
    }
    env = GridWorldRLlibEnv(lambda: FakeInnerEnv(num_agents=2), inequity_kwargs=inequity_kwargs)
    env.reset(seed=0)
    env._inequity_reward.trace["agent-0"] = 42.0
    env.reset(seed=1)
    assert env._inequity_reward.trace["agent-0"] == 0.0
    assert isinstance(env._env.rng, np.random.Generator)


SMALL_CONFIG = GridWorldConfig(height=12, width=16, episode_length=5)


@pytest.mark.parametrize("factory", [make_cleanup_rllib_env, make_harvest_rllib_env])
def test_real_env_factories_produce_valid_obs_and_step(factory):
    env = factory({"num_agents": 2, "env_config": SMALL_CONFIG, "seed": 0})
    obs, _infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)
    for aid in env.possible_agents:
        o = obs[aid]
        assert o.shape == env.observation_space.shape
        assert o.dtype == np.float32
        assert (o >= 0.0).all() and (o <= 1.0).all()

    action_dict = {aid: 0 for aid in env.possible_agents}
    obs2, rewards, terminations, truncations, _infos = env.step(action_dict)
    assert set(rewards) == set(env.possible_agents)
    assert terminations["__all__"] is False
    assert truncations["__all__"] is False
    for aid in env.possible_agents:
        assert obs2[aid].shape == env.observation_space.shape


def test_observe_trace_false_leaves_obs_space_and_shape_unchanged():
    env = _fake_env()
    assert env.observation_space.shape == (3 * OBS_HW * OBS_HW,)
    obs, _infos = env.reset(seed=0)
    assert obs["agent-0"].shape == (3 * OBS_HW * OBS_HW,)


def test_observe_trace_true_extends_obs_space_with_unbounded_trace_dims():
    env = GridWorldRLlibEnv(lambda: FakeInnerEnv(num_agents=2), observe_trace=True)
    image_dim = 3 * OBS_HW * OBS_HW
    assert env.observation_space.shape == (image_dim + 2,)
    # Image portion stays [0, 1]-bounded; trace portion is unbounded (the
    # trace is an unnormalized accumulator that can exceed 1 or go
    # negative -- see hughes2018/reward/inequity_aversion.py's docstring).
    assert np.all(env.observation_space.low[:image_dim] == 0.0)
    assert np.all(env.observation_space.high[:image_dim] == 1.0)
    assert np.all(env.observation_space.low[image_dim:] == -np.inf)
    assert np.all(env.observation_space.high[image_dim:] == np.inf)


def test_observe_trace_true_with_no_inequity_kwargs_builds_no_tracker_object():
    # Regression test for a real bug caught in review: an earlier version
    # built a live-updating InequityAversionReward(alpha=0, beta=0) here
    # "for architectural uniformity." Its .trace is computed by apply()'s
    # decay*trace + r, which runs *independent* of alpha/beta -- so even
    # at alpha=beta=0 it's a real, informative smoothed-reward signal, not
    # a no-op, giving a no-inequity-kwargs condition (baseline) a real
    # observation channel the paper's own unmodified A3C baseline never
    # had. _inequity_reward must stay None here; see _trace_vector() for
    # how the observation still gets a (true, constant-zero) trace slice.
    env = GridWorldRLlibEnv(lambda: FakeInnerEnv(num_agents=2), observe_trace=True)
    assert env._inequity_reward is None


def test_observe_trace_true_with_no_inequity_kwargs_is_a_true_constant_zero_not_a_live_tracker():
    # Regression test for the same bug: even after steps with nonzero
    # reward, the trace slice must stay exactly zero when there's no real
    # InequityAversionReward configured -- not track FakeInnerEnv's
    # reward_value=1.0 the way a live alpha=beta=0 tracker would.
    image_dim = 3 * OBS_HW * OBS_HW
    env = GridWorldRLlibEnv(lambda: FakeInnerEnv(num_agents=2), observe_trace=True)
    env.reset(seed=0)
    action_dict = {aid: 0 for aid in env.possible_agents}
    for _ in range(3):
        obs, _rewards, *_ = env.step(action_dict)
        assert np.allclose(obs["agent-0"][image_dim:], 0.0)
        assert np.allclose(obs["agent-1"][image_dim:], 0.0)


def test_observe_trace_reset_appends_zero_trace_step_appends_post_apply_trace():
    image_dim = 3 * OBS_HW * OBS_HW
    inequity_kwargs = {"alpha": {"agent-0": 0.5, "agent-1": 0.5}, "beta": {"agent-0": 0.5, "agent-1": 0.5}}
    env = GridWorldRLlibEnv(
        lambda: FakeInnerEnv(num_agents=2), inequity_kwargs=inequity_kwargs, observe_trace=True
    )
    obs, _infos = env.reset(seed=0)
    # Freshly reset -> trace is zero for every agent.
    assert np.allclose(obs["agent-0"][image_dim:], 0.0)
    assert np.allclose(obs["agent-1"][image_dim:], 0.0)

    obs2, _rewards, *_ = env.step({aid: 0 for aid in env.possible_agents})
    # FakeInnerEnv.step() gives reward_value=1.0 to every agent; decay =
    # gamma*trace_lambda = 0.99*0.95 = 0.9405 (InequityAversionReward's
    # defaults) -> trace after this step = 0.9405*0.0 + 1.0 = 1.0. The
    # *returned* observation (for the *next* action) should carry that
    # freshly-updated value, not the pre-step zero.
    assert np.allclose(obs2["agent-0"][image_dim:], [1.0, 1.0])
    assert np.allclose(obs2["agent-1"][image_dim:], [1.0, 1.0])


def test_no_env_config_uses_each_envs_own_default_map_size_not_a_generic_one():
    # Regression test for a bug caught in review: the factories used to fall
    # back to a bare GridWorldConfig() (18x25) when no env_config was
    # passed, silently overriding CleanupEnv's real default (18x26) and
    # HarvestEnv's (16x22) with the wrong map size and never regrowing/
    # cleaning in a paper-faithful proportion. Passing env_config=None
    # through to the inner env class -- not defaulting it in the wrapper --
    # is what lets each env apply its own correct default.
    from hughes2018.envs.cleanup import CleanupEnv
    from hughes2018.envs.harvest import HarvestEnv

    cleanup_env = make_cleanup_rllib_env({"num_agents": 2, "seed": 0})
    assert (cleanup_env._env.cfg.height, cleanup_env._env.cfg.width) == (
        CleanupEnv(num_agents=2).cfg.height,
        CleanupEnv(num_agents=2).cfg.width,
    )

    harvest_env = make_harvest_rllib_env({"num_agents": 2, "seed": 0})
    assert (harvest_env._env.cfg.height, harvest_env._env.cfg.width) == (
        HarvestEnv(num_agents=2).cfg.height,
        HarvestEnv(num_agents=2).cfg.width,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
