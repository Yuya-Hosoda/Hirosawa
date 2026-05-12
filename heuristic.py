"""
Energy-aware heuristic function h(v, b) — Section 4.2.2 (Eqs. 4.33–4.44).
Uses BFS-precomputed distances and LRU cache for efficiency.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple
from config import SimConfig
from grid_map import GridMap


class EnergyAwareHeuristic:
    """Heuristic h(v, b) for a specific goal position.

    Implements Eq. 4.44:
        h(v, b) = min{ h_direct(v, b), h_cs(v, b) }
    """

    def __init__(self, grid_map: GridMap, goal_x: int, goal_y: int,
                 config: SimConfig):
        self.grid_map = grid_map
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.config = config

        # Precompute BFS distances from goal (Eq. 4.35)
        self.dist_to_goal = grid_map.bfs_distance(
            goal_x, goal_y, config.agent_radius
        )

        # Precompute distances to nearest CS (Eq. 4.34)
        self.dist_to_cs = grid_map.compute_cs_distance_map(config.agent_radius)

        # Cache keyed by (x, y, b_level_ceil)
        self._cache: dict = {}

    def compute(self, x: int, y: int, b: float) -> float:
        """Compute h(v, b) = min{ h_direct, h_cs } (Eq. 4.44).

        Uses ceiling-based discretization for caching so that
        we never underestimate the agent's battery in heuristic computation.
        """
        cfg = self.config
        delta = cfg.soc_delta

        # Cache key: ceiling-based level
        b_level = math.ceil(b / delta) if delta > 0 else int(b)
        key = (x, y, b_level)
        if key in self._cache:
            return self._cache[key]

        # Effective battery = upper bound of bucket (optimistic for admissibility)
        b_eff = b_level * delta

        # d_goal(v): BFS distance to goal (Eq. 4.35)
        d_goal = self.dist_to_goal[y, x]
        if d_goal == np.inf:
            self._cache[key] = float('inf')
            return float('inf')

        t_goal = d_goal * cfg.cost_move        # Eq. 4.35
        e_goal = d_goal * cfg.energy_move      # Eq. 4.36
        e_need = e_goal + cfg.goal_min_battery  # Eq. 4.39

        # h_direct (Eq. 4.40)
        h_direct = t_goal if b_eff >= e_need else float('inf')

        # h_cs: via nearest CS (Eq. 4.41)
        d_cs = self.dist_to_cs[y, x]
        if d_cs == np.inf:
            h_cs = float('inf')
        else:
            t_cs = d_cs * cfg.cost_move
            e_cs = d_cs * cfg.energy_move
            b_at_cs = b_eff - e_cs

            if b_at_cs < 0:
                h_cs = float('inf')
            else:
                if b_at_cs >= e_need:
                    t_charge = 0.0
                else:
                    t_charge = (e_need - b_at_cs) / cfg.avg_charge_rate
                h_cs = t_cs + t_charge + t_goal  # Eq. 4.41 (simplified)

        result = min(h_direct, h_cs)
        self._cache[key] = result
        return result

    def is_reachable(self, x: int, y: int) -> bool:
        return self.dist_to_goal[y, x] < np.inf
