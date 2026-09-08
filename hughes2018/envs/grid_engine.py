"""Shared multi-agent gridworld engine for Cleanup and Harvest.

Both environments in Hughes et al. (2018), "Inequity Aversion Improves
Cooperation in Intertemporal Social Dilemmas" (NeurIPS 2018), share the same
low-level mechanics (the same mechanics later ported by Vinitsky et al.'s
`sequential_social_dilemma_games` and reused by McKee et al. 2023's Clean Up):
agents occupy grid cells, face one of 4 cardinal directions, move relative to
their own facing (forward/backward/strafe-left/strafe-right, not the absolute
compass direction), and observe an agent-centered, orientation-corrected
RGB crop of the map. A 3-cell-wide beam (a "punishment"/"fire" beam in both
envs, plus a "cleaning" beam in Cleanup only) fires forward from the agent.

This is a from-scratch reimplementation of that mechanic, independent of
Vinitsky et al.'s code (see the top-level README for why that distinction
matters) -- but it necessarily converges on the same movement/beam/
observation conventions, since those aren't an implementation choice, they're
what the paper's own environments do (and what any correct implementation of
them looks like). The square-crop-then-rotate observation technique below
follows the same proof technique validated in the sibling Leibo2017 repo's
`grid_utils.py` (a square array's center is a fixed point of 90-degree
rotation, so no rotation-target coordinate bookkeeping is needed) --
parameterized differently here for this paper's symmetric view radius and
3-wide beam, rather than Leibo2017's forward-biased window and 1-wide beam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
DIRS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])  # (drow, dcol) per orientation

# The paper's shared action set (Materials and Methods): move
# forward/backward/strafe-left/strafe-right (relative to current facing,
# not the absolute compass direction), turn left/right, stay, and a beam.
# Cleanup adds a second, cleaning beam action on top of this base set.
STEP_FORWARD, STEP_BACKWARD, STEP_LEFT, STEP_RIGHT = 0, 1, 2, 3
TURN_LEFT, TURN_RIGHT, STAY, FIRE = 4, 5, 6, 7
CLEAN = 8  # Cleanup-only

NUM_ACTIONS_BASE = 8  # Harvest's action space size
NUM_ACTIONS_CLEANUP = 9  # Harvest's actions + CLEAN

# DeepMind's reference implementation of this environment
# (github.com/google-deepmind/lab2d/tree/main/dmlab2d/lib/game_scripts/levels/clean_up,
# avatar.lua's `fineWait`/`cleanWait`) rate-limits both beams -- an agent
# that fires must wait this many steps before firing that same beam type
# again. Not mentioned in the paper's own prose, found only by reading the
# reference code; not previously implemented here, so agents could fire
# every step. FIRE_COOLDOWN_STEPS is shared by both games (the paper: "all
# agents are equipped with a fining beam" in both Cleanup and Harvest);
# CLEAN_COOLDOWN_STEPS is Cleanup-only, set via GridWorldEnv.clean_cooldown_steps.
FIRE_COOLDOWN_STEPS = 10

# Cell types.
EMPTY, WALL, APPLE, WASTE, RIVER, SPAWN = 0, 1, 2, 3, 4, 5

# Fire/clean beam: 3 cells wide (center + one cell either side of the firing
# line), 5 cells long, matching CleanupEnv/HarvestEnv's own FIRE length and
# default beam_width=3 (social_dilemmas/envs/cleanup.py's
# `_CLEANUP_ACTIONS = {"FIRE": 5, "CLEAN": 5}`, `map_env.py`'s
# `update_map_fire(..., beam_width=3)`).
BEAM_RANGE = 5
BEAM_WIDTH = 3

# Colors for rendering the grid to RGB before cropping.
COLOR_EMPTY = (0, 0, 0)
COLOR_WALL = (128, 128, 128)
COLOR_APPLE = (0, 255, 0)
COLOR_WASTE = (99, 156, 194)
COLOR_RIVER = (113, 75, 24)
COLOR_SELF = (0, 0, 255)
COLOR_OTHER = (255, 0, 0)
COLOR_FIRE_BEAM = (255, 255, 0)
COLOR_CLEAN_BEAM = (100, 255, 255)

_CELL_COLOR = {
    EMPTY: COLOR_EMPTY,
    WALL: COLOR_WALL,
    APPLE: COLOR_APPLE,
    WASTE: COLOR_WASTE,
    RIVER: COLOR_RIVER,
    SPAWN: COLOR_EMPTY,
}


def move_delta(orientation: int, action: int) -> tuple[int, int]:
    """Row/col delta for a movement action, relative to facing `orientation`."""
    if action == STEP_FORWARD:
        d = DIRS[orientation]
    elif action == STEP_BACKWARD:
        d = -DIRS[orientation]
    elif action == STEP_LEFT:
        d = DIRS[(orientation - 1) % 4]
    elif action == STEP_RIGHT:
        d = DIRS[(orientation + 1) % 4]
    else:
        return 0, 0
    return int(d[0]), int(d[1])


def rotate_orientation(orientation: int, action: int) -> int:
    if action == TURN_LEFT:
        return (orientation - 1) % 4
    if action == TURN_RIGHT:
        return (orientation + 1) % 4
    return orientation


def beam_lines(row: int, col: int, orientation: int, grid_shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    """The 3 unobstructed lines of a BEAM_WIDTH-wide, BEAM_RANGE-long beam.

    Returns one list of cells per line (center, then the two side lines),
    each in firing order, up to BEAM_RANGE cells or the grid edge -- NOT
    stopped at walls/agents/targets, since that blocking behavior differs by
    beam type (FIRE stops at the first wall or agent; Cleanup's CLEAN beam
    stops at the first waste cell it converts). A caller must walk each
    returned line independently and stop at its own first blocking cell;
    each line is independent (a caller must not break out of the whole
    beam on a per-line stop condition -- one line being blocked doesn't
    block the other two). The center line's first cell is one cell ahead of
    the agent; each side line's first cell is the cell immediately beside
    the agent (not diagonal -- an earlier version added the perpendicular
    offset without also stepping back by `fwd` first, so side lines started
    one cell diagonally forward of the agent and ran one cell short at the
    far end; caught in review against `map_env.py::update_map_fire`'s
    `start_pos + right_shift - firing_direction` construction, which is
    exactly this "step back by fwd before offsetting" correction).
    """
    fwd = DIRS[orientation]
    perp = DIRS[(orientation + 1) % 4]  # perpendicular ("right" of firing direction)
    h, w = grid_shape
    lines = []
    for offset in (0, 1, -1):
        start_r = row + perp[0] * offset - (fwd[0] if offset != 0 else 0)
        start_c = col + perp[1] * offset - (fwd[1] if offset != 0 else 0)
        line = []
        r, c = start_r, start_c
        for _ in range(BEAM_RANGE):
            r, c = r + fwd[0], c + fwd[1]
            if not (0 <= r < h and 0 <= c < w):
                break
            line.append((r, c))
        lines.append(line)
    return lines


_VIEW_RADIUS_DEFAULT = 7  # -> 15x15 window, matching CleanupEnv/HarvestEnv's own view size


def local_observation(rgb: np.ndarray, row: int, col: int, orientation: int, view_radius: int = _VIEW_RADIUS_DEFAULT) -> np.ndarray:
    """Crop + orient a symmetric agent-centered observation window.

    `rgb` has shape (H, W, 3). Returns shape (3, 2*view_radius+1, 2*view_radius+1):
    the agent always "faces up" in its own observation. See module docstring
    for why the square-crop-then-rotate technique needs no rotation-target
    coordinate bookkeeping.
    """
    h, w, _ = rgb.shape
    pad = view_radius
    padded = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=rgb.dtype)
    padded[pad:pad + h, pad:pad + w] = rgb
    pr, pc = row + pad, col + pad
    square = padded[pr - pad:pr + pad + 1, pc - pad:pc + pad + 1]
    rotated = np.rot90(square, k=orientation, axes=(0, 1))
    return np.transpose(rotated, (2, 0, 1)).copy()  # (3, H, W)


@dataclass
class GridAgent:
    agent_id: str
    row: int
    col: int
    orientation: int
    reward_this_step: float = 0.0
    fire_cooldown: int = 0  # steps remaining before FIRE is usable again
    clean_cooldown: int = 0  # steps remaining before CLEAN is usable again


@dataclass
class GridWorldConfig:
    height: int = 18
    width: int = 25
    episode_length: int = 1000
    view_radius: int = _VIEW_RADIUS_DEFAULT


class GridWorldEnv:
    """Base engine shared by CleanupEnv and HarvestEnv.

    A subclass provides: `_static_grid()` (the walls/base layout, called
    once), `_spawn_points()` (candidate agent start cells), `_reset_cells()`
    (place the env-specific resource, e.g. apples or waste, at episode
    start), `_custom_action(agent, action) -> reward` (handle any
    environment-specific action beyond the shared move/turn/stay/fire set,
    returning the extrinsic reward for taking it), and `_map_update()`
    (per-step resource dynamics, e.g. apple regrowth). This mirrors the hook
    shape of Vinitsky's `MapEnv` (a sensible design for this class of
    environment, since both papers' own environments share the same generic
    step loop) but is a fresh implementation, not a port of it.
    """

    num_actions: int = NUM_ACTIONS_BASE
    fire_cooldown_steps: int = FIRE_COOLDOWN_STEPS
    clean_cooldown_steps: int = 0  # no CLEAN action in the base engine; CleanupEnv overrides this

    def __init__(self, num_agents: int, config: GridWorldConfig | None = None, rng: np.random.Generator | None = None):
        self.num_agents = num_agents
        self.cfg = config or GridWorldConfig()
        self.rng = rng or np.random.default_rng()
        self._static_grid_arr = self._static_grid()
        self.reset()

    # -- hooks for subclasses -------------------------------------------------
    def _static_grid(self) -> np.ndarray:
        raise NotImplementedError

    def _spawn_points(self) -> list[tuple[int, int]]:
        raise NotImplementedError

    def _reset_cells(self, grid: np.ndarray) -> None:
        raise NotImplementedError

    def _custom_action(self, agent: GridAgent, action: int) -> float:
        return 0.0

    def _map_update(self, grid: np.ndarray) -> None:
        pass

    # -- generic engine ---------------------------------------------------
    def reset(self) -> dict[str, np.ndarray]:
        self._t = 0
        self.grid = self._static_grid_arr.copy()
        self._reset_cells(self.grid)
        spawn_points = list(self._spawn_points())
        self.rng.shuffle(spawn_points)
        self.agents: dict[str, GridAgent] = {}
        for i in range(self.num_agents):
            r, c = spawn_points[i % len(spawn_points)]
            orientation = int(self.rng.integers(4))
            self.agents[f"agent-{i}"] = GridAgent(f"agent-{i}", r, c, orientation)
        self._last_beam_cells: list[tuple[int, int, int]] = []  # (row, col, color_id) for render() only
        return self._observations()

    def _occupied_cells(self) -> set[tuple[int, int]]:
        return {(a.row, a.col) for a in self.agents.values()}

    def step(self, actions: dict[str, int]):
        for agent in self.agents.values():
            agent.reward_this_step = 0.0

        order = list(self.agents.keys())
        self.rng.shuffle(order)

        # Movement + rotation, resolved in random per-step order: an agent
        # moves into its target cell only if unoccupied (by another
        # not-yet-moved agent or one that already moved there this step) and
        # not a wall; otherwise it stays put. This is a simpler, documented
        # alternative to Vinitsky's multi-pass swap-resolution algorithm
        # (which additionally lets two agents trade places in one step) --
        # see the README's "What's simplified vs. the paper" section.
        occupied = self._occupied_cells()
        for agent_id in order:
            agent = self.agents[agent_id]
            action = actions.get(agent_id, STAY)
            if action in (TURN_LEFT, TURN_RIGHT):
                agent.orientation = rotate_orientation(agent.orientation, action)
            elif action in (STEP_FORWARD, STEP_BACKWARD, STEP_LEFT, STEP_RIGHT):
                dr, dc = move_delta(agent.orientation, action)
                nr, nc = agent.row + dr, agent.col + dc
                if (
                    0 <= nr < self.grid.shape[0]
                    and 0 <= nc < self.grid.shape[1]
                    and self.grid[nr, nc] != WALL
                    and (nr, nc) not in occupied
                ):
                    occupied.discard((agent.row, agent.col))
                    agent.row, agent.col = nr, nc
                    occupied.add((nr, nc))

        # Consume the cell the agent now stands on (e.g. an apple).
        for agent in self.agents.values():
            if self.grid[agent.row, agent.col] == APPLE:
                agent.reward_this_step += 1.0
                self.grid[agent.row, agent.col] = EMPTY

        # Beams (FIRE, and CLEAN for Cleanup), random order for the same
        # reason movement is randomized (no fixed-agent-index priority).
        # Each beam type has its own independent cooldown counter, ticked
        # down every step regardless of which action was actually taken
        # this step (matching the reference implementation's semantics --
        # see FIRE_COOLDOWN_STEPS) -- so an agent that fires must wait
        # `fire_cooldown_steps`/`clean_cooldown_steps` steps before that
        # same beam type is usable again.
        beam_order = list(order)
        self.rng.shuffle(beam_order)
        self._last_beam_cells = []
        for agent_id in beam_order:
            agent = self.agents[agent_id]
            action = actions.get(agent_id, STAY)

            if agent.fire_cooldown > 0:
                agent.fire_cooldown -= 1
            elif action == FIRE:
                agent.fire_cooldown = self.fire_cooldown_steps
                self._fire_beam(agent)

            if agent.clean_cooldown > 0:
                agent.clean_cooldown -= 1
            elif action == CLEAN:
                agent.clean_cooldown = self.clean_cooldown_steps
                agent.reward_this_step += self._custom_action(agent, CLEAN)

            if action >= NUM_ACTIONS_BASE and action != CLEAN:
                # Non-beam custom actions (currently unused, reserved for
                # subclasses that add more than one extra action beyond
                # CLEAN -- CLEAN itself is already handled above). Deliberately
                # its own independent check, not chained onto clean_cooldown's
                # if/elif: an earlier version chained it there, so whenever an
                # agent happened to be on CLEAN's cooldown, any *other*,
                # unrelated custom action would be silently skipped too --
                # dormant today (no subclass defines one yet) but a real bug
                # for whichever subclass adds the first one.
                agent.reward_this_step += self._custom_action(agent, action)

        self._map_update(self.grid)

        self._t += 1
        done = self._t >= self.cfg.episode_length
        observations = self._observations()
        rewards = {aid: a.reward_this_step for aid, a in self.agents.items()}
        dones = {aid: done for aid in self.agents}
        dones["__all__"] = done
        infos = {aid: {} for aid in self.agents}
        return observations, rewards, dones, infos

    def _fire_beam(self, agent: GridAgent) -> None:
        """The punishment beam is a fine, not a timeout/removal (Hughes et
        al. 2018 explicitly contrasts this with the earlier SSD literature's
        timeout-based punishment beam -- caught in review against the
        primary text; an earlier version of this method incorrectly
        implemented timeout-based removal instead)."""
        lines = beam_lines(agent.row, agent.col, agent.orientation, self.grid.shape)
        agent.reward_this_step -= 1.0  # cost of firing (paper: -1 for the shooter)
        hit_cells = []
        pos_to_agent = {(a.row, a.col): a for a in self.agents.values() if a is not agent}
        for line in lines:
            for (r, c) in line:
                if self.grid[r, c] == WALL:
                    break  # stops only this line, not the other two
                hit_cells.append((r, c))
                hit_agent = pos_to_agent.get((r, c))
                if hit_agent is not None:
                    hit_agent.reward_this_step -= 50.0  # the fine
                    break  # this line stops at the agent it hits
        self._last_beam_cells += [(r, c, 0) for (r, c) in hit_cells]

    def _render_rgb(self, viewer_id: str | None) -> np.ndarray:
        h, w = self.grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cell_type, color in _CELL_COLOR.items():
            rgb[self.grid == cell_type] = color
        for agent_id, agent in self.agents.items():
            color = COLOR_SELF if agent_id == viewer_id else COLOR_OTHER
            rgb[agent.row, agent.col] = color
        return rgb

    def _observations(self) -> dict[str, np.ndarray]:
        return {
            agent_id: local_observation(self._render_rgb(agent_id), agent.row, agent.col, agent.orientation, self.cfg.view_radius)
            for agent_id, agent in self.agents.items()
        }

    def render(self) -> np.ndarray:
        """Full-map RGB frame (H, W, 3) for third-person visualization."""
        rgb = self._render_rgb(None).copy()
        for (r, c, _color_id) in self._last_beam_cells:
            rgb[r, c] = COLOR_FIRE_BEAM
        return rgb
