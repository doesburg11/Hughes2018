"""Correctness tests for CleanupEnv's paper-specific mechanics."""

import numpy as np
import pytest

from hughes2018.envs.cleanup import CleanupEnv
from hughes2018.envs.grid_engine import (
    APPLE,
    CLEAN,
    NORTH,
    RIVER,
    STAY,
    WASTE,
    GridWorldConfig,
)


def _make_env(num_agents=2, seed=0):
    rng = np.random.default_rng(seed)
    return CleanupEnv(num_agents=num_agents, config=GridWorldConfig(episode_length=1000), rng=rng)


def test_river_starts_fully_polluted_and_orchard_empty():
    env = _make_env()
    assert all(env.grid[r, c] == WASTE for (r, c) in env.river_cells)
    assert all(env.grid[r, c] != APPLE for (r, c) in env.apple_cells)


def test_clean_action_converts_first_waste_cell_on_the_beam():
    env = _make_env(num_agents=1)
    agent = env.agents["agent-0"]
    # Face the agent at a known waste cell, one cell south of it.
    target_r, target_c = env.river_cells[0]
    agent.row, agent.col, agent.orientation = target_r + 1, target_c, NORTH
    assert env.grid[target_r, target_c] == WASTE
    env.step({"agent-0": CLEAN})
    assert env.grid[target_r, target_c] == RIVER


def test_apple_spawn_probability_is_zero_above_depletion_threshold():
    env = _make_env(num_agents=1)
    env._compute_spawn_probabilities(env.grid)  # river is 100% waste at reset
    assert env.current_apple_spawn_prob == 0.0
    assert env.current_waste_spawn_prob == 0.0  # also stops spawning more waste once saturated


def test_apple_spawn_probability_is_maximal_when_river_is_clean():
    env = _make_env(num_agents=1)
    for (r, c) in env.river_cells:
        env.grid[r, c] = RIVER
    env._compute_spawn_probabilities(env.grid)
    assert env.current_apple_spawn_prob == pytest.approx(0.05)


def test_apple_spawn_probability_interpolates_between_thresholds():
    env = _make_env(num_agents=1)
    # Half the river cleaned -> waste_density = 0.5 * THRESHOLD_DEPLETION,
    # i.e. halfway through the interpolation range -> half the max apple
    # spawn probability.
    half = len(env.river_cells) // 2
    for (r, c) in env.river_cells[:half]:
        env.grid[r, c] = RIVER
    waste_density = 1 - (half / len(env.river_cells))
    env._compute_spawn_probabilities(env.grid)
    if waste_density < 0.4:  # THRESHOLD_DEPLETION
        expected = (1 - waste_density / 0.4) * 0.05
        assert env.current_apple_spawn_prob == pytest.approx(expected)


def test_cleaning_reduces_waste_density_over_many_steps():
    env = _make_env(num_agents=2, seed=1)
    agents = list(env.agents.values())
    for i, (r, c) in enumerate(env.river_cells[: len(agents)]):
        agents[i].row, agents[i].col = r + 1, c
        agents[i].orientation = NORTH
    initial_waste = int(np.count_nonzero(env.grid == WASTE))
    for _ in range(50):
        env.step({aid: CLEAN for aid in env.agents})
        for i, (r, c) in enumerate(env.river_cells[: len(agents)]):
            agents[i].row, agents[i].col = r + 1, c
    final_waste = int(np.count_nonzero(env.grid == WASTE))
    assert final_waste < initial_waste


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
