"""Correctness tests for the shared gridworld primitives (grid_engine.py)."""

import numpy as np
import pytest

from hughes2018.envs.grid_engine import (
    APPLE,
    EMPTY,
    FIRE,
    NORTH,
    EAST,
    SOUTH,
    WEST,
    STAY,
    STEP_BACKWARD,
    STEP_FORWARD,
    STEP_LEFT,
    STEP_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    WALL,
    GridWorldConfig,
    GridWorldEnv,
    beam_lines,
    local_observation,
    move_delta,
    rotate_orientation,
)


def test_move_delta_forward_matches_facing():
    assert move_delta(NORTH, STEP_FORWARD) == (-1, 0)
    assert move_delta(EAST, STEP_FORWARD) == (0, 1)
    assert move_delta(SOUTH, STEP_FORWARD) == (1, 0)
    assert move_delta(WEST, STEP_FORWARD) == (0, -1)


def test_move_delta_backward_is_opposite_of_forward():
    for orientation in (NORTH, EAST, SOUTH, WEST):
        fwd = move_delta(orientation, STEP_FORWARD)
        back = move_delta(orientation, STEP_BACKWARD)
        assert back == (-fwd[0], -fwd[1])


def test_move_delta_strafe_is_perpendicular_to_forward():
    for orientation in (NORTH, EAST, SOUTH, WEST):
        fwd = move_delta(orientation, STEP_FORWARD)
        left = move_delta(orientation, STEP_LEFT)
        right = move_delta(orientation, STEP_RIGHT)
        # A 2D dot product of zero confirms perpendicularity.
        assert fwd[0] * left[0] + fwd[1] * left[1] == 0
        assert fwd[0] * right[0] + fwd[1] * right[1] == 0
        assert left == (-right[0], -right[1])


def test_rotate_orientation_cycles_through_all_four():
    o = NORTH
    for _ in range(4):
        o = rotate_orientation(o, TURN_RIGHT)
    assert o == NORTH
    o = NORTH
    for _ in range(4):
        o = rotate_orientation(o, TURN_LEFT)
    assert o == NORTH
    assert rotate_orientation(NORTH, TURN_RIGHT) == EAST
    assert rotate_orientation(NORTH, TURN_LEFT) == WEST


def test_beam_lines_center_line_starts_one_cell_ahead():
    lines = beam_lines(row=5, col=5, orientation=NORTH, grid_shape=(20, 20))
    center_line = lines[0]
    assert center_line[0] == (4, 5)  # one cell north of (5, 5)
    assert len(center_line) == 5  # BEAM_RANGE

    lines_east = beam_lines(row=5, col=5, orientation=EAST, grid_shape=(20, 20))
    assert lines_east[0][0] == (5, 6)


def test_beam_lines_side_lines_are_offset_perpendicular():
    lines = beam_lines(row=5, col=5, orientation=NORTH, grid_shape=(20, 20))
    _center, side_a, side_b = lines
    # Facing NORTH, perpendicular ("right") is EAST -> side lines start at
    # column 6 and column 4.
    assert side_a[0][1] == 6
    assert side_b[0][1] == 4


def test_beam_lines_truncated_at_grid_edge():
    lines = beam_lines(row=1, col=5, orientation=NORTH, grid_shape=(20, 20))
    assert len(lines[0]) == 1  # only row 0 is reachable before the edge


def test_local_observation_places_agent_at_center():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    obs = local_observation(rgb, row=3, col=4, orientation=NORTH, view_radius=2)
    assert obs.shape == (3, 5, 5)  # (C, 2*radius+1, 2*radius+1)


def test_local_observation_forward_direction_is_always_up():
    # Place a unique marker one cell "north" of the agent (row-1) and verify
    # it lands in the row above center after local_observation, for every
    # orientation -- i.e. "forward" is always up in the agent's own view,
    # regardless of which way it's actually facing on the grid.
    for orientation, marker_pos in (
        (NORTH, (4, 5)),  # forward = north = row-1
        (EAST, (5, 6)),  # forward = east = col+1
        (SOUTH, (6, 5)),  # forward = south = row+1
        (WEST, (5, 4)),  # forward = west = col-1
    ):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[marker_pos] = (255, 255, 255)
        obs = local_observation(rgb, row=5, col=5, orientation=orientation, view_radius=3)
        center = 3
        assert tuple(obs[:, center - 1, center]) == (255, 255, 255), orientation


class _DummyEnv(GridWorldEnv):
    """Minimal concrete subclass for exercising the generic step loop."""

    num_actions = 8

    def _static_grid(self):
        grid = np.full((10, 10), EMPTY, dtype=np.int8)
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL
        return grid

    def _spawn_points(self):
        return [(r, c) for r in range(1, 9) for c in range(1, 9)]

    def _reset_cells(self, grid):
        grid[3, 3] = APPLE


def test_dummy_env_all_actions_step_without_error():
    env = _DummyEnv(num_agents=2, config=GridWorldConfig(height=10, width=10, episode_length=5))
    for action in range(8):
        env.step({"agent-0": action, "agent-1": STAY})


def test_agent_cannot_walk_through_walls():
    env = _DummyEnv(num_agents=1, config=GridWorldConfig(height=10, width=10, episode_length=100))
    agent = env.agents["agent-0"]
    agent.row, agent.col, agent.orientation = 1, 1, NORTH
    env.step({"agent-0": STEP_FORWARD})
    assert (agent.row, agent.col) == (1, 1)  # blocked by the wall at row 0


def test_apple_pickup_gives_reward_and_clears_cell():
    env = _DummyEnv(num_agents=1, config=GridWorldConfig(height=10, width=10, episode_length=100))
    agent = env.agents["agent-0"]
    agent.row, agent.col, agent.orientation = 3, 2, NORTH  # one cell left of the apple at (3, 3)
    _obs, rewards, _dones, _infos = env.step({"agent-0": STEP_RIGHT})
    assert agent.row == 3 and agent.col == 3
    assert rewards["agent-0"] == 1.0
    assert env.grid[3, 3] == EMPTY


def test_fire_beam_costs_shooter_and_removes_target():
    env = _DummyEnv(num_agents=2, config=GridWorldConfig(height=10, width=10, episode_length=100))
    shooter, target = env.agents["agent-0"], env.agents["agent-1"]
    shooter.row, shooter.col, shooter.orientation = 5, 5, NORTH
    target.row, target.col = 3, 5  # directly ahead, within BEAM_RANGE=5
    _obs, rewards, _dones, _infos = env.step({"agent-0": FIRE, "agent-1": STAY})
    assert rewards["agent-0"] == -1.0
    assert rewards["agent-1"] == -50.0
    assert target.removed_timer == env.cfg.removal_steps


def test_removed_agent_is_excluded_from_occupied_cells_and_observations():
    env = _DummyEnv(num_agents=2, config=GridWorldConfig(height=10, width=10, episode_length=100))
    target = env.agents["agent-1"]
    target.removed_timer = 5
    obs, _rewards, _dones, _infos = env.step({"agent-0": STAY, "agent-1": STAY})
    assert "agent-1" in obs  # a removed agent still receives an observation/turn
    assert target.removed_timer == 4  # cooldown ticks down


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
