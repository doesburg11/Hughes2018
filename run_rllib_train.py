#!/usr/bin/env python3
# ruff: noqa: E402
"""Train Cleanup/Harvest through Ray RLlib instead of the default from-scratch
single-process path (hughes2018/training/loop.py). See
hughes2018/envs/rllib_wrappers.py's module docstring for why this exists:
RLlib already solves N-parallel-env-runner training reliably, which this
repo's own hand-rolled multiprocessing attempt (abandoned mid-build after
hitting a real bug on the first test) would otherwise be reinventing.

    pip install -r requirements-rllib.txt   # ray[rllib]; not in the base requirements.txt

    # Cleanup, IMPALA (closest RLlib algorithm to the paper's own
    # asynchronous actor-critic with staleness correction -- V-trace),
    # 8 parallel env-runners, advantageous-only inequity aversion:
    python run_rllib_train.py --env cleanup --algorithm IMPALA \
        --num-env-runners 8 --condition advantageous_only --iterations 200

    # Baseline, no inequity term:
    python run_rllib_train.py --env cleanup --condition baseline --iterations 200
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms.impala import IMPALA, IMPALAConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hughes2018.envs.rllib_wrappers import make_cleanup_rllib_env, make_harvest_rllib_env
from hughes2018.reward.inequity_aversion import scaled_alpha_beta_range

ENV_FACTORIES = {"cleanup": make_cleanup_rllib_env, "harvest": make_harvest_rllib_env}


def build_inequity_kwargs(condition: str, agent_ids: list[str], trace_lambda: float, seed: int) -> dict | None:
    """condition: "baseline" | "advantageous_only" | "both" -- see
    run_experiment1_cleanup.py's module docstring for why these three
    (matching the paper's Fig. 3, which tests advantageous-only and
    disadvantageous-only separately, never combined -- "both" is this
    repo's own broader condition, kept for reference)."""
    if condition == "baseline":
        return None
    rng = np.random.default_rng(seed + 1000)
    alpha_range, beta_range = scaled_alpha_beta_range(gamma=0.99, trace_lambda=trace_lambda)
    if condition == "advantageous_only":
        alpha = {aid: 0.0 for aid in agent_ids}
    elif condition == "both":
        alpha = {aid: float(rng.uniform(*alpha_range)) for aid in agent_ids}
    elif condition == "disadvantageous_only":
        alpha = {aid: float(rng.uniform(*alpha_range)) for aid in agent_ids}
        beta = {aid: 0.0 for aid in agent_ids}
        return {"alpha": alpha, "beta": beta, "trace_lambda": trace_lambda}
    else:
        raise ValueError(f"Unknown condition: {condition!r}")
    beta = {aid: float(rng.uniform(*beta_range)) for aid in agent_ids}
    return {"alpha": alpha, "beta": beta, "trace_lambda": trace_lambda}


def get_algorithm(name: str):
    name = name.upper()
    if name == "PPO":
        return PPO, PPOConfig()
    if name == "IMPALA":
        return IMPALA, IMPALAConfig()
    raise ValueError(f"Unsupported algorithm '{name}'. Supported: PPO, IMPALA.")


def build_env_config(env_name: str, num_agents: int, episode_length: int):
    """`env_name`'s own default map size (CleanupEnv=18x26, HarvestEnv=16x22),
    discovered from the env class itself via a throwaway probe instance
    rather than duplicated here as magic numbers, with only
    `episode_length` overridden. (Caught in review: an earlier version of
    this script always built a bare `GridWorldConfig(episode_length=...)`,
    which silently trained on the *generic* 18x25 default instead of
    either env's real default map.)"""
    from hughes2018.envs.grid_engine import GridWorldConfig

    if env_name == "cleanup":
        from hughes2018.envs.cleanup import CleanupEnv as _EnvClass
    elif env_name == "harvest":
        from hughes2018.envs.harvest import HarvestEnv as _EnvClass
    else:
        raise ValueError(f"Unknown env '{env_name}'")

    default_cfg = _EnvClass(num_agents=num_agents).cfg
    return GridWorldConfig(height=default_cfg.height, width=default_cfg.width, episode_length=episode_length)


def env_seed_for_worker(base_seed: int, worker_index: int, vector_index: int) -> int:
    """Give each parallel RLlib env-runner (`worker_index`) and each
    vectorized sub-env within it (`vector_index`) a distinct env seed, so
    parallel rollout workers produce genuinely different trajectories
    instead of N identical copies of the same environment realization.
    (Caught in review: an earlier version of this script always passed the
    same `--seed` to every worker via `env_creator`, ignoring RLlib's
    `EnvContext` entirely -- confirmed to produce identical reset
    observations across workers.)"""
    return base_seed + worker_index * 1000 + vector_index


def resolve_checkpoint_dir(checkpoint_dir: str | None) -> str | None:
    """None passes through; otherwise resolves to an absolute path.

    algo.save() resolves its `checkpoint_dir` via
    `pyarrow.fs.FileSystem.from_uri()`, which requires an absolute path or
    an explicit URI scheme -- a relative path fails with `ArrowInvalid: URI
    has empty scheme` only at save time, i.e. only after a full training
    run has already completed. Caught in review after this silently lost
    two full paper-scale training runs' checkpoints (the training itself
    succeeded both times; only the final save() call failed)."""
    if checkpoint_dir is None:
        return None
    return str(Path(checkpoint_dir).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_FACTORIES), default="cleanup")
    parser.add_argument("--algorithm", choices=["PPO", "IMPALA"], default="IMPALA")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument(
        "--condition",
        choices=["baseline", "advantageous_only", "disadvantageous_only", "both"],
        default="baseline",
    )
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--inequity-trace-lambda", type=float, default=0.95)
    parser.add_argument("--num-env-runners", type=int, default=8, help="Parallel workers -- tune to your CPU count.")
    parser.add_argument("--num-envs-per-env-runner", type=int, default=1)
    parser.add_argument(
        "--rollout-length", type=int, default=20, help="LSTM truncated-BPTT window (RLlib's max_seq_len)."
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--gpu", action="store_true", help="Train the (tiny) network on GPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--observe-trace",
        dest="observe_trace",
        action="store_true",
        default=True,
        help="Agents observe every player's inequity-trace vector (Sec. 3.2). On by default.",
    )
    parser.add_argument(
        "--no-observe-trace",
        dest="observe_trace",
        action="store_false",
        help="Disable trace observability -- inequity aversion (if any) still shapes the reward, just invisibly.",
    )
    args = parser.parse_args()
    args.checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)

    ray.init(include_dashboard=False)

    env_config = build_env_config(args.env, args.num_agents, args.episode_length)
    agent_ids = [f"agent-{i}" for i in range(args.num_agents)]
    inequity_kwargs = build_inequity_kwargs(args.condition, agent_ids, args.inequity_trace_lambda, args.seed)

    env_name = f"{args.env}_rllib_env"

    def env_creator(env_ctx):
        # env_ctx is RLlib's EnvContext (a dict subclass) when called from a
        # real env-runner, but a plain {} when called directly (e.g. this
        # script's own probe_env below) -- getattr's default covers both.
        worker_index = getattr(env_ctx, "worker_index", 0) or 0
        vector_index = getattr(env_ctx, "vector_index", 0) or 0
        return ENV_FACTORIES[args.env](
            {
                "num_agents": args.num_agents,
                "env_config": env_config,
                "inequity_kwargs": inequity_kwargs,
                "observe_trace": args.observe_trace,
                "seed": env_seed_for_worker(args.seed, worker_index, vector_index),
            }
        )

    register_env(env_name, env_creator)

    probe_env = env_creator({})
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space

    policy_specs = {aid: PolicySpec(observation_space=obs_space, action_space=act_space) for aid in agent_ids}

    def policy_mapping_fn(agent_id, *_args, **_kwargs):
        return agent_id

    algo_cls, config = get_algorithm(args.algorithm)
    config = (
        config.framework("torch")
        .environment(env=env_name)
        .env_runners(num_env_runners=args.num_env_runners, num_envs_per_env_runner=args.num_envs_per_env_runner)
        .resources(num_gpus=1 if args.gpu else 0)
        .multi_agent(policies=policy_specs, policy_mapping_fn=policy_mapping_fn)
        .rl_module(
            model_config={
                "use_lstm": True,
                "lstm_cell_size": 128,
                "max_seq_len": args.rollout_length,
                "fcnet_hiddens": [64, 64],
                # No conv_filters: observations are pre-flattened by
                # GridWorldRLlibEnv (see its module docstring for why).
            }
        )
        .training(lr=args.lr, entropy_coeff=args.entropy_coeff, gamma=0.99)
    )

    algo = None
    try:
        algo = algo_cls(config=config)
        trace_label = "trace-observing" if args.observe_trace else "trace-blind"
        print(
            f"=== Training {args.env} / {args.condition} / {args.algorithm} / "
            f"{args.num_env_runners} env-runners / {trace_label} ==="
        )
        for i in range(args.iterations):
            result = algo.train()
            env_runner_stats = result.get("env_runners", {})
            return_mean = env_runner_stats.get("episode_return_mean")
            agent_returns = env_runner_stats.get("agent_episode_returns_mean", {})
            steps = result.get("num_env_steps_sampled_lifetime")
            print(f"iter {i:4d}  steps={steps:>8}  episode_return_mean={return_mean}  per_agent={agent_returns}")

        if args.checkpoint_dir:
            path = algo.save(checkpoint_dir=args.checkpoint_dir)
            print(f"Saved checkpoint to {path}")
    finally:
        # Always release Ray actors, even on a training/checkpoint exception
        # -- an earlier version only did this on the happy path, which could
        # leave env-runner/learner actors alive in notebooks, tests, or
        # embedded runs (caught in review).
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
