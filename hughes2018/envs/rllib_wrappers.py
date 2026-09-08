"""Optional Ray RLlib backend for CleanupEnv/HarvestEnv.

This is a separate, additive training path, not a replacement for
`hughes2018/agents/actor_critic.py` and `hughes2018/training/loop.py`
(those remain the default, from-scratch, single-process training path this
repo is built around -- see the top-level README's disambiguation from the
sibling SequentialSocialDilemmas repo, which *is* an RLlib port). This
module exists for one specific reason: this repo's own hand-rolled
multi-process training loop is exactly the kind of infrastructure RLlib
already solves (N parallel env-runners, per-agent independent policies,
gradient/weight synchronization) -- reinventing and debugging that here
would be duplicating well-tested work for no benefit over just using it,
the same reasoning the sibling Leibo2017 repo gives for its own optional
RLlib backend. Requires `ray[rllib]`, which is *not* in `requirements.txt`
-- see `requirements-rllib.txt`.

The inequity-aversion reward is applied inside this wrapper's `step()`
(not baked into CleanupEnv/HarvestEnv themselves, which stay reward-
mechanism-agnostic, matching `training/loop.py`'s external-application
design) -- exactly the same place `training/loop.py` applies it in the
single-process path.

**Documented simplification vs. the custom path**: observations here are
flattened to a 1D Box rather than kept as a (C, H, W) image, and the model
config below relies on `fcnet_hiddens` + `use_lstm` only, no explicit conv
layer -- RLlib's Catalog auto-detects 3D Box observations as images and has
no preset for this environment's non-standard channel-first shape (the
same issue the sibling SequentialSocialDilemmas repo's own RLlib adapter
documents and sidesteps the same way). `hughes2018/agents/actor_critic.py`'s
own network *does* include the paper-matching 3x3/32-channel conv layer;
this RLlib path is a further, RLlib-specific simplification on top of the
already-documented ones, not a claim that this matches the custom path's
architecture exactly.

`observe_trace=True` appends every agent's current inequity-trace value
(Sec. 3.2's "we allow agents to observe the smoothed reward of every player
on each timestep") to that same flat vector -- trivial here precisely
*because* the observation is already flattened: no second input branch is
needed the way `ActorCriticNetwork` needs one for its (C, H, W) input (see
that module for the causality rule this depends on getting right).
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from hughes2018.reward.inequity_aversion import InequityAversionReward


class GridWorldRLlibEnv(MultiAgentEnv):
    """Wraps CleanupEnv/HarvestEnv's plain dict-keyed reset()/step() API for
    RLlib's MultiAgentEnv contract (new API stack, terminated/truncated
    split). The underlying env's only "done" condition is reaching
    `episode_length` -- a time-limit truncation, not a true terminal state,
    so it's reported as `truncations`, not `terminations` (this lets
    RLlib's value bootstrapping treat it correctly, unlike this repo's own
    single-process `training/loop.py`, which -- documented there as a
    known simplification -- treats it as a true termination instead).
    """

    def __init__(
        self,
        inner_env_factory,
        inequity_kwargs: dict | None = None,
        observe_trace: bool = False,
    ):
        super().__init__()
        self._env = inner_env_factory()
        self.possible_agents = list(self._env.agents.keys())
        self.agents = list(self.possible_agents)
        self._observe_trace = observe_trace

        self._inequity_reward: InequityAversionReward | None = None
        if inequity_kwargs is not None:
            self._inequity_reward = InequityAversionReward(agent_ids=self.possible_agents, **inequity_kwargs)
        # No zero-alpha/beta tracker for the no-inequity-kwargs case (an
        # earlier version built one "for architectural uniformity"): its
        # `.trace` is computed by apply()'s `decay*trace + r`, which runs
        # *independent* of alpha/beta -- so even at alpha=beta=0 it's a
        # real, live-updating smoothed-reward signal, not a no-op. That
        # gave a condition with no inequity_kwargs (baseline) a real,
        # informative observation channel the paper's own unmodified A3C
        # baseline never had access to (Sec. 3.2's "we allow agents to
        # observe..." describes the inequity-averse agent's own design,
        # not the baseline it's compared against) -- confirmed as a live
        # bug when baseline consistently outperformed advantageous_only
        # across a multi-seed comparison, the opposite of the paper's own
        # result, because baseline was getting that information for free
        # while advantageous_only paid a real penalty for the same access.
        # See _convert_obs()/reset()/step() below: when observe_trace=True
        # and there's no real InequityAversionReward, the trace slice is a
        # true constant zero -- carries no information, matching
        # hughes2018/training/loop.py's already-correct baseline handling.

        obs_hw = 2 * self._env.cfg.view_radius + 1
        image_dim = 3 * obs_hw * obs_hw
        if observe_trace:
            trace_dim = len(self.possible_agents)
            # Unlike the [0, 1]-normalized image portion, the trace is an
            # unnormalized discounted accumulator (module docstring in
            # hughes2018/reward/inequity_aversion.py) that can run well
            # outside [0, 1] and can go negative (e.g. after a punishment
            # fine) -- bounding it to [0, 1] like the image would be wrong.
            low = np.concatenate([np.zeros(image_dim, dtype=np.float32), np.full(trace_dim, -np.inf, dtype=np.float32)])
            high = np.concatenate([np.ones(image_dim, dtype=np.float32), np.full(trace_dim, np.inf, dtype=np.float32)])
            obs_space = Box(low=low, high=high, dtype=np.float32)
        else:
            obs_space = Box(low=0.0, high=1.0, shape=(image_dim,), dtype=np.float32)
        act_space = Discrete(self._env.num_actions)
        self.observation_spaces = {aid: obs_space for aid in self.possible_agents}
        self.action_spaces = {aid: act_space for aid in self.possible_agents}
        self.observation_space = obs_space
        self.action_space = act_space

    def _convert_obs(
        self, obs_dict: dict[str, np.ndarray], trace_vector: list[float] | None
    ) -> dict[str, np.ndarray]:
        converted = {}
        for aid, o in obs_dict.items():
            flat = o.reshape(-1).astype(np.float32) / 255.0
            if trace_vector is not None:
                flat = np.concatenate([flat, np.asarray(trace_vector, dtype=np.float32)])
            converted[aid] = flat
        return converted

    def _trace_vector(self) -> list[float] | None:
        """The trace slice for the current observation, or None if this env
        wasn't built with observe_trace=True. Real trace when an actual
        InequityAversionReward is configured; a true constant zero
        (informationless) otherwise -- see __init__'s comment for why this
        distinction matters."""
        if not self._observe_trace:
            return None
        if self._inequity_reward is not None:
            return self._inequity_reward.observable_trace()
        return [0.0] * len(self.possible_agents)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)
        if self._inequity_reward is not None:
            self._inequity_reward.reset()
        obs = self._env.reset()
        # Trace is freshly zeroed by reset() above -- matches
        # training/loop.py's t=0 (no history yet) case.
        trace_vector = self._trace_vector()
        return self._convert_obs(obs, trace_vector), {aid: {} for aid in obs}

    def step(self, action_dict: dict[str, int]):
        obs, raw_rewards, dones, infos = self._env.step(action_dict)
        rewards = self._inequity_reward.apply(raw_rewards) if self._inequity_reward is not None else raw_rewards
        # apply() just updated the trace using *this* step's reward -- read
        # it now, so the *returned* (next) observation carries this step's
        # trace, for the action *after* the one that produced it. See
        # InequityAversionReward.observable_trace()'s docstring, case (b).
        trace_vector = self._trace_vector()

        all_done = bool(dones.get("__all__", False))
        terminations = {aid: False for aid in obs}
        truncations = {aid: all_done for aid in obs}
        terminations["__all__"] = False
        truncations["__all__"] = all_done

        return self._convert_obs(obs, trace_vector), rewards, terminations, truncations, infos

    def render(self) -> np.ndarray:
        return self._env.render()


def make_cleanup_rllib_env(config: dict | None = None) -> GridWorldRLlibEnv:
    from hughes2018.envs.cleanup import CleanupEnv

    config = dict(config or {})
    seed = config.pop("seed", None)
    inequity_kwargs = config.pop("inequity_kwargs", None)
    observe_trace = config.pop("observe_trace", False)
    num_agents = config.pop("num_agents", 5)
    # `env_config=None` here (the default) is intentional: CleanupEnv's own
    # __init__ then applies *its* correct default map size (18x26), not a
    # generic GridWorldConfig()'s (18x25) -- passing a bare GridWorldConfig()
    # from this wrapper silently trained a different map. See
    # run_rllib_train.py's `build_env_config()` for how a caller overrides
    # just `episode_length` without losing the env-specific default size.
    env_config = config.pop("env_config", None)

    def factory():
        return CleanupEnv(num_agents=num_agents, config=env_config, rng=np.random.default_rng(seed))

    return GridWorldRLlibEnv(factory, inequity_kwargs=inequity_kwargs, observe_trace=observe_trace)


def make_harvest_rllib_env(config: dict | None = None) -> GridWorldRLlibEnv:
    from hughes2018.envs.harvest import HarvestEnv

    config = dict(config or {})
    seed = config.pop("seed", None)
    inequity_kwargs = config.pop("inequity_kwargs", None)
    observe_trace = config.pop("observe_trace", False)
    num_agents = config.pop("num_agents", 5)
    # See make_cleanup_rllib_env's comment above -- same reasoning, HarvestEnv's
    # own default map size is 16x22.
    env_config = config.pop("env_config", None)

    def factory():
        return HarvestEnv(num_agents=num_agents, config=env_config, rng=np.random.default_rng(seed))

    return GridWorldRLlibEnv(factory, inequity_kwargs=inequity_kwargs, observe_trace=observe_trace)
