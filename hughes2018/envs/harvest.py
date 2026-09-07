"""Harvest: Hughes et al. (2018)'s tragedy-of-the-commons dilemma.

Apples regrow faster where more apples already stand nearby, so harvesting
too aggressively collapses the local regrowth rate for everyone, including
the harvester. Regrowth probability is a function of the count of apples in
a Moore (3x3) neighborhood around a given empty cell -- 0 nearby apples means
that cell can never regrow (permanent local extinction is possible), more
nearby apples means faster regrowth. `SPAWN_PROBABILITY_BY_NEIGHBOR_COUNT`
below is the paper's own stated regrowth curve (also independently
transcribed in Vinitsky et al.'s port, since it's the paper's mechanic, not
an implementation detail specific to either repo).
"""

from __future__ import annotations

import numpy as np

from hughes2018.envs.grid_engine import (
    APPLE,
    EMPTY,
    NUM_ACTIONS_BASE,
    WALL,
    GridWorldConfig,
    GridWorldEnv,
)

# Indexed by min(nearby_apple_count, 3): 0 neighbors -> can't regrow, 1/2/3+
# neighbors -> increasingly likely to regrow this step.
SPAWN_PROBABILITY_BY_NEIGHBOR_COUNT = (0.0, 0.005, 0.02, 0.05)
NEIGHBOR_OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]


class HarvestEnv(GridWorldEnv):
    num_actions = NUM_ACTIONS_BASE

    def __init__(self, num_agents: int = 5, config: GridWorldConfig | None = None, rng=None):
        self.apple_cells: list[tuple[int, int]] = []
        super().__init__(num_agents, config or GridWorldConfig(height=16, width=22), rng)

    def _static_grid(self) -> np.ndarray:
        cfg = self.cfg
        grid = np.full((cfg.height, cfg.width), EMPTY, dtype=np.int8)
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL
        self.apple_cells = [
            (r, c) for r in range(1, cfg.height - 1) for c in range(1, cfg.width - 1)
        ]
        return grid

    def _spawn_points(self) -> list[tuple[int, int]]:
        return self.apple_cells

    def _reset_cells(self, grid: np.ndarray) -> None:
        # The orchard starts full, matching the paper's own training
        # initialization (and the sibling repos' shared convention of
        # starting Harvest fully stocked).
        for (r, c) in self.apple_cells:
            grid[r, c] = APPLE

    def _map_update(self, grid: np.ndarray) -> None:
        occupied = {(a.row, a.col) for a in self.agents.values()}
        h, w = grid.shape
        new_apples = []
        candidates = list(self.apple_cells)
        self.rng.shuffle(candidates)
        for (r, c) in candidates:
            if grid[r, c] != EMPTY or (r, c) in occupied:
                continue
            neighbor_count = 0
            for (dr, dc) in NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == APPLE:
                    neighbor_count += 1
            prob = SPAWN_PROBABILITY_BY_NEIGHBOR_COUNT[min(neighbor_count, 3)]
            if prob > 0.0 and self.rng.random() < prob:
                new_apples.append((r, c))
        for (r, c) in new_apples:
            grid[r, c] = APPLE
