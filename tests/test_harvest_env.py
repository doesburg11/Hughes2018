"""Correctness tests for HarvestEnv's density-dependent regrowth."""

import numpy as np
import pytest

from hughes2018.envs.grid_engine import APPLE, EMPTY, GridWorldConfig
from hughes2018.envs.harvest import SPAWN_PROBABILITY_BY_NEIGHBOR_COUNT, HarvestEnv


def _make_env(num_agents=1, seed=0):
    rng = np.random.default_rng(seed)
    return HarvestEnv(num_agents=num_agents, config=GridWorldConfig(episode_length=1000), rng=rng)


def test_orchard_starts_full():
    env = _make_env()
    assert all(env.grid[r, c] == APPLE for (r, c) in env.apple_cells)


def test_isolated_empty_cell_never_regrows():
    env = _make_env(num_agents=0)
    r, c = 8, 8
    # Re-clear the Moore neighborhood every iteration, not just once: apples
    # from outside the block can spread inward over many steps, eventually
    # giving (r, c) nonzero neighbors indirectly even though it started
    # isolated -- the point of this test is the *local* 0-neighbor rule, so
    # the neighborhood must stay empty at the moment each check runs.
    for _ in range(200):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                env.grid[r + dr, c + dc] = EMPTY
        env._map_update(env.grid)
        assert env.grid[r, c] == EMPTY  # 0-neighbor probability is exactly 0.0


def test_dense_neighborhood_regrows_eventually():
    env = _make_env(num_agents=0, seed=2)
    r, c = 8, 8
    env.grid[r, c] = EMPTY  # every neighbor stays APPLE -> 8 neighbors -> max spawn prob
    regrew = False
    for _ in range(500):
        env._map_update(env.grid)
        if env.grid[r, c] == APPLE:
            regrew = True
            break
    assert regrew


def test_spawn_probability_table_is_nondecreasing():
    probs = SPAWN_PROBABILITY_BY_NEIGHBOR_COUNT
    assert probs == tuple(sorted(probs))
    assert probs[0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
