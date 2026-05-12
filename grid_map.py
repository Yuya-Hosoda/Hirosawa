"""
Grid map representation and BFS distance precomputation.
Implements the environment model G=(V,E) from Section 4.1.1.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import List, Tuple, Set, Optional, Dict


# Cell types
CELL_FREE = 0
CELL_OBSTACLE = 1
CELL_CS = 2


class GridMap:
    """2D grid graph with obstacles and charging stations."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.cs_positions: List[Tuple[int, int]] = []   # (x, y) list
        self.cs_capacity: Dict[Tuple[int, int], int] = {}

    def set_obstacle(self, x: int, y: int):
        self.grid[y, x] = CELL_OBSTACLE

    def set_cs(self, x: int, y: int, capacity: int = 1):
        self.grid[y, x] = CELL_CS
        pos = (x, y)
        if pos not in self.cs_positions:
            self.cs_positions.append(pos)
        self.cs_capacity[pos] = capacity

    def is_free(self, x: int, y: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] != CELL_OBSTACLE
        return False

    def is_cs(self, x: int, y: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == CELL_CS
        return False

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def can_agent_occupy(self, cx: int, cy: int, agent_radius: int = 1) -> bool:
        """Check if a 3x3 agent centered at (cx, cy) can occupy this position.
        All cells in the (2*r+1) x (2*r+1) area must be free or CS."""
        r = agent_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if not self.is_in_bounds(nx, ny):
                    return False
                if self.grid[ny, nx] == CELL_OBSTACLE:
                    return False
        return True

    def get_4neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """4-connected neighbors."""
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if self.is_in_bounds(nx, ny) and self.grid[ny, nx] != CELL_OBSTACLE:
                neighbors.append((nx, ny))
        return neighbors

    def bfs_distance(self, goal_x: int, goal_y: int,
                     agent_radius: int = 1) -> np.ndarray:
        """BFS from goal to all reachable cells, considering agent radius.
        Returns distance array (inf for unreachable cells).
        Used for d_goal(v) in Eq. 4.35 and d_cs(v) in Eq. 4.34."""
        dist = np.full((self.height, self.width), np.inf)
        if not self.can_agent_occupy(goal_x, goal_y, agent_radius):
            return dist

        dist[goal_y, goal_x] = 0
        queue = deque([(goal_x, goal_y)])

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (self.is_in_bounds(nx, ny) and
                        dist[ny, nx] == np.inf and
                        self.can_agent_occupy(nx, ny, agent_radius)):
                    dist[ny, nx] = dist[y, x] + 1
                    queue.append((nx, ny))
        return dist

    def compute_cs_distance_map(self, agent_radius: int = 1) -> np.ndarray:
        """Compute d_cs(v) = min distance from each cell to nearest CS (Eq. 4.34).
        Multi-source BFS from all CS positions."""
        dist = np.full((self.height, self.width), np.inf)
        queue = deque()

        for cx, cy in self.cs_positions:
            if self.can_agent_occupy(cx, cy, agent_radius):
                dist[cy, cx] = 0
                queue.append((cx, cy))

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (self.is_in_bounds(nx, ny) and
                        dist[ny, nx] == np.inf and
                        self.can_agent_occupy(nx, ny, agent_radius)):
                    dist[ny, nx] = dist[y, x] + 1
                    queue.append((nx, ny))
        return dist


def create_scenario1() -> GridMap:
    """Scenario1 from Fig. 5.1: 30×30, 3 CS, L-shaped obstacle.
    CS dispersed across the map as in the paper."""
    gm = GridMap(30, 30)

    # L-shaped obstacle
    for x in range(10, 20):
        for y in range(12, 18):
            gm.set_obstacle(x, y)
    for x in range(10, 15):
        for y in range(8, 12):
            gm.set_obstacle(x, y)

    # 3 CS matching Fig 5.1
    gm.set_cs(15, 2, capacity=1)   # top-center
    gm.set_cs(26, 14, capacity=1)  # right
    gm.set_cs(14, 27, capacity=1)  # bottom-center

    return gm


def create_empty_map(width: int, height: int,
                     cs_positions: List[Tuple[int, int]]) -> GridMap:
    """Create an empty map with given CS positions."""
    gm = GridMap(width, height)
    for x, y in cs_positions:
        gm.set_cs(x, y, capacity=1)
    return gm
