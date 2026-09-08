"""Cleanup: Hughes et al. (2018)'s public-goods intertemporal social dilemma.

Apples give reward but only regrow in proportion to how clean a shared river
is; cleaning the river costs travel time away from the orchard and yields no
direct reward. An agent that never cleans free-rides on whoever does.

Apple regrowth vs. river pollution follows the paper's own piecewise-linear
relationship (also transcribed, independently, in Vinitsky et al.'s port and
in McKee et al. 2023's reuse of this same environment -- these are the
paper's stated dynamics, not an implementation detail borrowed from either):
apples stop regrowing entirely once waste covers >= `THRESHOLD_DEPLETION` of
the river's potential waste area, and regrow at the maximum rate once waste
coverage is at `THRESHOLD_RESTORATION` (0), linearly interpolated between.

Map layout (own design, not a copy of any existing implementation's ASCII
map -- see the top-level README's "What's simplified vs. the paper" section):
a river region on the left, an orchard on the right, connected by an open
floor strip where agents spawn. The paper doesn't publish its exact map
layout either (same "blind spot" the sibling Leibo2017 repo documents for
Gathering's map), so any reasonable map with a spatially separate river and
orchard is a faithful reading.
"""

from __future__ import annotations

import numpy as np

from hughes2018.envs.grid_engine import (
    APPLE,
    CLEAN,
    EMPTY,
    NUM_ACTIONS_CLEANUP,
    RIVER,
    WALL,
    WASTE,
    GridWorldConfig,
    GridWorldEnv,
    beam_lines,
)

THRESHOLD_DEPLETION = 0.4
THRESHOLD_RESTORATION = 0.0
WASTE_SPAWN_PROBABILITY = 0.5
# The paper's own Sec. A.3 prose reads "apples spawn in the field with
# probability 0.125x" (x = normalized saturation), which an earlier version
# of this env took literally. But DeepMind's own maintained reference
# implementation of this exact level
# (github.com/google-deepmind/lab2d/tree/main/dmlab2d/lib/game_scripts/levels/clean_up,
# simulation.lua's `appleRespawnProbability` default) uses 0.05, and agrees
# with the paper's text on every *other* checkable constant here
# (thresholdDepletion=0.4, thresholdRestoration=0.0, mudSpawnProbability=0.5)
# -- code is less error-prone than a hand-written sentence, so 0.05 is
# better-evidenced than the paper's own prose on this one number. (Not to be
# confused with Melting Pot's own `clean_up` substrate, a *different*,
# later, 7-player, timeout-beam variant that only shares the name/citation
# with this paper -- not used as evidence here.)
APPLE_RESPAWN_PROBABILITY = 0.05

# Hughes et al. (2018), Sec. 2.4: "At the start of each episode, the
# environment resets with waste just beyond this saturation point" -- i.e.
# just past THRESHOLD_DEPLETION, not fully saturated. An earlier version of
# this env filled the *entire* river with waste at reset (density 1.0),
# requiring far more cleaning before any apple could possibly spawn than the
# paper's own design; caught by reading the primary source directly. The
# paper doesn't give an exact margin past the threshold, so 0.42 (a modest,
# documented margin above 0.4) is this repo's own interpretation of "just
# beyond", not a verified reproduction of a specific unstated number.
RESET_WASTE_DENSITY = 0.42

# DeepMind's reference implementation (see APPLE_RESPAWN_PROBABILITY's
# comment for the source) delays new waste accumulation for the first 50
# steps of each episode (`dirtGrowthStartTime`) -- not mentioned in the
# paper's own prose, found only by reading the reference code. Only new
# waste spawning is paused; the episode's initial waste (RESET_WASTE_DENSITY)
# is unaffected.
DIRT_GROWTH_START_TIME = 50


class CleanupEnv(GridWorldEnv):
    num_actions = NUM_ACTIONS_CLEANUP
    clean_cooldown_steps = 2  # DeepMind's reference implementation's `cleanWait`; see FIRE_COOLDOWN_STEPS's comment

    def __init__(self, num_agents: int = 5, config: GridWorldConfig | None = None, rng=None):
        self.river_cells: list[tuple[int, int]] = []
        self.apple_cells: list[tuple[int, int]] = []
        self.spawn_cells: list[tuple[int, int]] = []
        super().__init__(num_agents, config or GridWorldConfig(height=18, width=26), rng)

    def _static_grid(self) -> np.ndarray:
        cfg = self.cfg
        grid = np.full((cfg.height, cfg.width), EMPTY, dtype=np.int8)
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        # River (left), orchard (right), and a spawn strip between them --
        # sized proportionally to cfg.width (roughly a quarter each for
        # river/orchard, the remainder for spawning) rather than fixed
        # column offsets, so a smaller custom GridWorldConfig (e.g. for a
        # fast test) doesn't silently produce an empty/inverted spawn zone.
        interior_width = cfg.width - 2  # excluding the two wall columns
        zone_width = max(3, interior_width // 4)
        min_interior = 2 * zone_width + 1  # river + orchard + >=1 spawn column
        if interior_width < min_interior:
            raise ValueError(
                f"cfg.width={cfg.width} is too narrow for CleanupEnv's river/spawn/orchard "
                f"layout (needs interior width >= {min_interior}, got {interior_width})"
            )
        river_cols = range(1, 1 + zone_width)
        apple_cols = range(cfg.width - 1 - zone_width, cfg.width - 1)
        spawn_cols = range(1 + zone_width, cfg.width - 1 - zone_width)
        rows = range(2, cfg.height - 2)
        if not rows:
            raise ValueError(f"cfg.height={cfg.height} is too short for CleanupEnv (needs >= 5)")

        self.river_cells = [(r, c) for r in rows for c in river_cols]
        self.apple_cells = [(r, c) for r in rows for c in apple_cols]
        self.spawn_cells = [(r, c) for r in rows for c in spawn_cols]
        return grid

    def _spawn_points(self) -> list[tuple[int, int]]:
        return self.spawn_cells

    def _reset_cells(self, grid: np.ndarray) -> None:
        # Waste starts at RESET_WASTE_DENSITY (see its docstring) -- just
        # beyond THRESHOLD_DEPLETION, not the whole river -- matching the
        # paper's own stated reset condition. The orchard starts empty
        # either way, since current_apple_spawn_prob starts at 0 regardless
        # (apples can't spawn above THRESHOLD_DEPLETION, which this reset
        # density is chosen to just exceed).
        num_waste_cells = round(RESET_WASTE_DENSITY * len(self.river_cells))
        cells = list(self.river_cells)
        self.rng.shuffle(cells)
        for (r, c) in cells[:num_waste_cells]:
            grid[r, c] = WASTE
        self.potential_waste_area = len(self.river_cells)
        self.current_apple_spawn_prob = 0.0
        self.current_waste_spawn_prob = WASTE_SPAWN_PROBABILITY

    def _custom_action(self, agent, action: int) -> float:
        if action != CLEAN:
            return 0.0
        lines = beam_lines(agent.row, agent.col, agent.orientation, self.grid.shape)
        for line in lines:
            for (r, c) in line:
                if self.grid[r, c] == WALL:
                    break
                if self.grid[r, c] == WASTE:
                    self.grid[r, c] = RIVER
                    break  # the cleaning beam stops at the first waste cell it clears
        return 0.0  # cleaning itself has no direct extrinsic reward

    def _map_update(self, grid: np.ndarray) -> None:
        self._compute_spawn_probabilities(grid)
        self._spawn_waste(grid)
        self._spawn_apples(grid)

    def _compute_spawn_probabilities(self, grid: np.ndarray) -> None:
        waste_count = int(np.count_nonzero(grid == WASTE))
        waste_density = waste_count / self.potential_waste_area if self.potential_waste_area else 0.0
        if waste_density >= THRESHOLD_DEPLETION:
            self.current_apple_spawn_prob = 0.0
            self.current_waste_spawn_prob = 0.0
        else:
            self.current_waste_spawn_prob = WASTE_SPAWN_PROBABILITY
            if waste_density <= THRESHOLD_RESTORATION:
                self.current_apple_spawn_prob = APPLE_RESPAWN_PROBABILITY
            else:
                span = THRESHOLD_DEPLETION - THRESHOLD_RESTORATION
                self.current_apple_spawn_prob = (
                    1 - (waste_density - THRESHOLD_RESTORATION) / span
                ) * APPLE_RESPAWN_PROBABILITY

    def _spawn_waste(self, grid: np.ndarray) -> None:
        if self._t <= DIRT_GROWTH_START_TIME:  # grace period; see DIRT_GROWTH_START_TIME
            return
        if np.isclose(self.current_waste_spawn_prob, 0.0):
            return
        candidates = [(r, c) for (r, c) in self.river_cells if grid[r, c] != WASTE]
        self.rng.shuffle(candidates)
        for (r, c) in candidates:
            if self.rng.random() < self.current_waste_spawn_prob:
                grid[r, c] = WASTE
                break  # only one new waste cell can spawn per step

    def _spawn_apples(self, grid: np.ndarray) -> None:
        if np.isclose(self.current_apple_spawn_prob, 0.0):
            return
        occupied = {(a.row, a.col) for a in self.agents.values()}
        for (r, c) in self.apple_cells:
            if grid[r, c] == EMPTY and (r, c) not in occupied and self.rng.random() < self.current_apple_spawn_prob:
                grid[r, c] = APPLE
