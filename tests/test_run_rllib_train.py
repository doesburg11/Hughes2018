"""Correctness test for run_rllib_train.py's build_inequity_kwargs wiring.

Mirrors tests/test_run_experiment1_cleanup.py's coverage of
run_experiment1_cleanup.py's build_inequity_reward, but for the RLlib
path's own build_inequity_kwargs -- which additionally supports
"disadvantageous_only" (not present in the non-RLlib script). Same failure
mode this guards against: a silently-wrong alpha/beta here would make a
condition test something other than what its name claims, without erroring.
"""

import sys
from pathlib import Path

import pytest

# ray[rllib] is an optional extra (requirements-rllib.txt), not part of the
# base install -- skip this whole module rather than fail collection when
# it's absent, so `pytest tests/ -q` still passes on a base install. Needed
# here even though this file's own tests don't call into Ray: importing
# run_rllib_train.py itself does `import ray` at module level.
pytest.importorskip("ray.rllib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_rllib_train import (  # noqa: E402
    build_env_config,
    build_inequity_kwargs,
    env_seed_for_worker,
    resolve_checkpoint_dir,
)


def _agent_ids():
    return [f"agent-{i}" for i in range(5)]


def test_baseline_returns_none():
    assert build_inequity_kwargs("baseline", _agent_ids(), trace_lambda=0.95, seed=0) is None


def test_advantageous_only_zeroes_alpha_but_keeps_beta_positive():
    kwargs = build_inequity_kwargs("advantageous_only", _agent_ids(), trace_lambda=0.95, seed=0)
    assert kwargs is not None
    assert all(v == 0.0 for v in kwargs["alpha"].values())
    assert all(v > 0.0 for v in kwargs["beta"].values())


def test_disadvantageous_only_zeroes_beta_but_keeps_alpha_positive():
    kwargs = build_inequity_kwargs("disadvantageous_only", _agent_ids(), trace_lambda=0.95, seed=0)
    assert kwargs is not None
    assert all(v > 0.0 for v in kwargs["alpha"].values())
    assert all(v == 0.0 for v in kwargs["beta"].values())


def test_both_samples_positive_alpha_and_beta():
    kwargs = build_inequity_kwargs("both", _agent_ids(), trace_lambda=0.95, seed=0)
    assert kwargs is not None
    assert all(v > 0.0 for v in kwargs["alpha"].values())
    assert all(v > 0.0 for v in kwargs["beta"].values())


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        build_inequity_kwargs("bogus", _agent_ids(), trace_lambda=0.95, seed=0)


def test_all_conditions_carry_agent_ids_for_every_agent():
    for condition in ("advantageous_only", "disadvantageous_only", "both"):
        kwargs = build_inequity_kwargs(condition, _agent_ids(), trace_lambda=0.95, seed=0)
        assert set(kwargs["alpha"]) == set(_agent_ids())
        assert set(kwargs["beta"]) == set(_agent_ids())


def test_env_seed_for_worker_differs_across_workers_and_vector_indices():
    # Regression test for a bug caught in review: an earlier version of
    # env_creator ignored RLlib's EnvContext entirely and passed the same
    # --seed to every parallel env-runner, so all of them replayed an
    # identical environment realization instead of contributing genuinely
    # diverse experience.
    base = 0
    seeds = {env_seed_for_worker(base, w, v) for w in range(4) for v in range(3)}
    assert len(seeds) == 12  # every (worker_index, vector_index) pair is distinct
    assert env_seed_for_worker(base, 0, 0) == base


def test_resolve_checkpoint_dir_passes_none_through():
    assert resolve_checkpoint_dir(None) is None


def test_resolve_checkpoint_dir_makes_a_relative_path_absolute(tmp_path, monkeypatch):
    # Regression test for a bug caught in review: algo.save() resolves its
    # checkpoint_dir via pyarrow.fs.FileSystem.from_uri(), which rejects a
    # relative path with "ArrowInvalid: URI has empty scheme" -- but only
    # at save time, after training has already finished. This silently
    # lost two full paper-scale training runs' checkpoints before being
    # caught. cwd is set to a known tmp_path so the expected absolute
    # path is exact, not just "looks absolute".
    monkeypatch.chdir(tmp_path)
    resolved = resolve_checkpoint_dir("output/checkpoints/run1")
    assert resolved == str(tmp_path / "output" / "checkpoints" / "run1")
    assert Path(resolved).is_absolute()


def test_resolve_checkpoint_dir_leaves_an_already_absolute_path_equivalent(tmp_path):
    absolute = str(tmp_path / "checkpoints")
    assert resolve_checkpoint_dir(absolute) == absolute


def test_build_env_config_uses_each_envs_own_default_map_size():
    # Regression test for a bug caught in review: this used to always build
    # a bare GridWorldConfig(episode_length=...), silently overriding
    # CleanupEnv's/HarvestEnv's real default map sizes with the generic
    # 18x25 default.
    from hughes2018.envs.cleanup import CleanupEnv
    from hughes2018.envs.harvest import HarvestEnv

    cleanup_cfg = build_env_config("cleanup", num_agents=2, episode_length=123)
    assert (cleanup_cfg.height, cleanup_cfg.width) == (
        CleanupEnv(num_agents=2).cfg.height,
        CleanupEnv(num_agents=2).cfg.width,
    )
    assert cleanup_cfg.episode_length == 123

    harvest_cfg = build_env_config("harvest", num_agents=2, episode_length=456)
    assert (harvest_cfg.height, harvest_cfg.width) == (
        HarvestEnv(num_agents=2).cfg.height,
        HarvestEnv(num_agents=2).cfg.width,
    )
    assert harvest_cfg.episode_length == 456


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
