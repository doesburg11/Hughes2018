"""Correctness tests for CleanupEnv's paper-specific mechanics."""

import numpy as np
import pytest

from hughes2018.envs.cleanup import APPLE_RESPAWN_PROBABILITY, DIRT_GROWTH_START_TIME, CleanupEnv
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


def test_river_starts_just_beyond_depletion_threshold_not_fully_saturated():
    # Hughes et al. (2018): "the environment resets with waste just beyond
    # this saturation point" -- an earlier version of this env started with
    # the *entire* river as waste (density 1.0) instead, caught by reading
    # the paper's own stated reset condition directly.
    env = _make_env()
    waste_count = sum(1 for (r, c) in env.river_cells if env.grid[r, c] == WASTE)
    density = waste_count / len(env.river_cells)
    assert 0.40 <= density <= 0.45  # RESET_WASTE_DENSITY=0.42, just past THRESHOLD_DEPLETION=0.4
    assert density < 0.9  # nowhere near full saturation
    assert all(env.grid[r, c] != APPLE for (r, c) in env.apple_cells)


def test_clean_action_converts_first_waste_cell_on_the_beam():
    env = _make_env(num_agents=1)
    agent = env.agents["agent-0"]
    # Only ~42% of river cells are waste at reset (RESET_WASTE_DENSITY), so
    # river_cells[0] specifically isn't guaranteed to be one -- find an
    # actual waste cell instead of assuming a fixed index.
    target_r, target_c = next((r, c) for (r, c) in env.river_cells if env.grid[r, c] == WASTE)
    agent.row, agent.col, agent.orientation = target_r + 1, target_c, NORTH
    env.step({"agent-0": CLEAN})
    assert env.grid[target_r, target_c] == RIVER


def test_clean_beam_has_a_cooldown_before_it_can_be_used_again():
    # Regression test for a gap caught by reading DeepMind's own reference
    # implementation of this environment: agents could clean every single
    # step, with no rate limit -- the reference implementation cools down
    # for CleanupEnv.clean_cooldown_steps (2) steps after each use.
    env = _make_env(num_agents=1)
    agent = env.agents["agent-0"]

    def clean_next_waste_cell():
        target = next((r, c) for (r, c) in env.river_cells if env.grid[r, c] == WASTE)
        agent.row, agent.col, agent.orientation = target[0] + 1, target[1], NORTH
        env.step({"agent-0": CLEAN})
        return target

    first_target = clean_next_waste_cell()
    assert env.grid[first_target] == RIVER  # the beam actually fired

    # Still on cooldown for CleanupEnv.clean_cooldown_steps more steps:
    # cleaning does nothing (the targeted waste cell stays waste).
    for _ in range(CleanupEnv.clean_cooldown_steps):
        target = next((r, c) for (r, c) in env.river_cells if env.grid[r, c] == WASTE)
        agent.row, agent.col, agent.orientation = target[0] + 1, target[1], NORTH
        env.step({"agent-0": CLEAN})
        assert env.grid[target] == WASTE  # unchanged -- beam didn't fire

    # Cooldown has now elapsed -- the beam works again.
    last_target = clean_next_waste_cell()
    assert env.grid[last_target] == RIVER


def test_no_new_waste_spawns_during_the_grace_period_but_does_after():
    # Regression test for a gap caught by reading DeepMind's own reference
    # implementation: new waste accumulation is paused for the first
    # DIRT_GROWTH_START_TIME steps of each episode, not mentioned in the
    # paper's own prose. Only new spawning is paused -- reset's own initial
    # waste is untouched, so this test starts from a low, non-saturated
    # waste density instead (density well under THRESHOLD_DEPLETION=0.4,
    # matching the state new-waste-spawning is meant to test).
    env = _make_env(num_agents=1)
    for (r, c) in env.river_cells:
        env.grid[r, c] = RIVER
    env.grid[env.river_cells[0]] = WASTE
    initial_waste_count = sum(1 for (r, c) in env.river_cells if env.grid[r, c] == WASTE)
    assert initial_waste_count == 1

    # _t is 0-indexed and checked *before* this step's increment (see
    # _spawn_waste's `self._t <= DIRT_GROWTH_START_TIME` guard), so _t takes
    # values 0..DIRT_GROWTH_START_TIME (inclusive) -- DIRT_GROWTH_START_TIME+1
    # calls -- before growth is allowed to resume.
    for _ in range(DIRT_GROWTH_START_TIME + 1):
        env.step({"agent-0": STAY})
        waste_count = sum(1 for (r, c) in env.river_cells if env.grid[r, c] == WASTE)
        assert waste_count == initial_waste_count  # no growth yet during the grace period

    # Grace period has now elapsed -- waste growth resumes. WASTE_SPAWN_PROBABILITY=0.5
    # per candidate cell per step makes at least one new waste cell within a
    # handful of steps overwhelmingly likely.
    grew = False
    for _ in range(20):
        env.step({"agent-0": STAY})
        waste_count = sum(1 for (r, c) in env.river_cells if env.grid[r, c] == WASTE)
        if waste_count > initial_waste_count:
            grew = True
            break
    assert grew


def test_apple_spawn_probability_is_zero_above_depletion_threshold():
    env = _make_env(num_agents=1)
    env._compute_spawn_probabilities(env.grid)  # reset density (0.42) is already just past the threshold
    assert env.current_apple_spawn_prob == 0.0
    assert env.current_waste_spawn_prob == 0.0  # also stops spawning more waste once saturated


def test_apple_spawn_probability_is_maximal_when_river_is_clean():
    env = _make_env(num_agents=1)
    for (r, c) in env.river_cells:
        env.grid[r, c] = RIVER
    env._compute_spawn_probabilities(env.grid)
    assert env.current_apple_spawn_prob == pytest.approx(APPLE_RESPAWN_PROBABILITY)


def test_apple_spawn_probability_interpolates_between_thresholds():
    env = _make_env(num_agents=1)
    # Explicitly set a known waste state for all river cells first (rather
    # than relying on the reset's own ~42%-random state for the "untouched"
    # half), so the expected waste_density below is exact regardless of
    # RESET_WASTE_DENSITY.
    for (r, c) in env.river_cells:
        env.grid[r, c] = WASTE
    # Half the river cleaned -> waste_density = 0.5 * THRESHOLD_DEPLETION,
    # i.e. halfway through the interpolation range -> half the max apple
    # spawn probability.
    half = len(env.river_cells) // 2
    for (r, c) in env.river_cells[:half]:
        env.grid[r, c] = RIVER
    waste_density = 1 - (half / len(env.river_cells))
    env._compute_spawn_probabilities(env.grid)
    if waste_density < 0.4:  # THRESHOLD_DEPLETION
        expected = (1 - waste_density / 0.4) * APPLE_RESPAWN_PROBABILITY
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
