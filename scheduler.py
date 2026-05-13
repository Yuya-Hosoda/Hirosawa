"""
Lifelong task scheduling, CS selection, and emergency intervention.
Sections 4.4.1–4.4.3. Properly tracks CS occupancy and queues.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

from config import SimConfig
from grid_map import GridMap
from agent import Agent


class CSQueue:
    """Track CS occupancy and reservation queues."""
    def __init__(self):
        # Agents heading toward or charging at each CS
        self.assigned: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        # Agent currently physically charging
        self.charging_at: Dict[Tuple[int, int], Optional[int]] = {}
        # History for stats
        self._q_sum: Dict[Tuple[int, int], float] = defaultdict(float)
        self._q_count: int = 0

    def assign_agent(self, cs_pos: Tuple[int, int], agent_id: int):
        """Agent is assigned to head toward / charge at this CS."""
        if agent_id not in self.assigned[cs_pos]:
            self.assigned[cs_pos].append(agent_id)

    def unassign_agent(self, cs_pos: Tuple[int, int], agent_id: int):
        if agent_id in self.assigned[cs_pos]:
            self.assigned[cs_pos].remove(agent_id)
        if self.charging_at.get(cs_pos) == agent_id:
            self.charging_at[cs_pos] = None

    def set_charging(self, cs_pos: Tuple[int, int], agent_id: int):
        self.charging_at[cs_pos] = agent_id

    def is_occupied(self, cs_pos: Tuple[int, int]) -> bool:
        return self.charging_at.get(cs_pos) is not None

    def get_queue_length(self, cs_pos: Tuple[int, int]) -> int:
        """Number of agents assigned to this CS (heading + charging)."""
        return len(self.assigned[cs_pos])

    def get_wait_count(self, cs_pos: Tuple[int, int]) -> int:
        """Agents that would have to wait (assigned minus capacity)."""
        return max(0, len(self.assigned[cs_pos]) - 1)

    def record(self):
        self._q_count += 1
        for cs_pos, agents in self.assigned.items():
            self._q_sum[cs_pos] += len(agents)

    def avg_queue_length(self, cs_pos: Tuple[int, int]) -> float:
        return self._q_sum[cs_pos] / self._q_count if self._q_count > 0 else 0.0

    def overall_avg(self) -> float:
        if self._q_count == 0 or not self._q_sum:
            return 0.0
        return sum(self._q_sum.values()) / (self._q_count * max(1, len(self._q_sum)))


class BFSCache:
    def __init__(self, grid_map: GridMap, agent_radius: int):
        self.grid_map = grid_map
        self.agent_radius = agent_radius
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._cs_dist: Optional[np.ndarray] = None

    def get_distance_from(self, x: int, y: int) -> np.ndarray:
        key = (x, y)
        if key not in self._cache:
            self._cache[key] = self.grid_map.bfs_distance(x, y, self.agent_radius)
        return self._cache[key]

    def get_cs_distance_map(self) -> np.ndarray:
        if self._cs_dist is None:
            self._cs_dist = self.grid_map.compute_cs_distance_map(self.agent_radius)
        return self._cs_dist


def compute_urgency(soc: float) -> float:
    """Urgency coefficient γ_urg from spec §11.1."""
    if soc < 0.10: return 5.0
    if soc < 0.20: return 3.0
    if soc < 0.30: return 1.5
    return 0.0


def select_cs_proposed(
    agent: Agent, grid_map: GridMap, cs_queue: CSQueue,
    config: SimConfig, bfs_cache: BFSCache,
) -> Optional[Tuple[int, int]]:
    """J-cost minimization (Eq. 4.57-4.59).
    Considers queue length, travel time, risk."""
    if not grid_map.cs_positions:
        return None

    dist_map = bfs_cache.get_distance_from(agent.x, agent.y)

    candidates = []
    for cs_pos in grid_map.cs_positions:
        d = dist_map[cs_pos[1], cs_pos[0]]
        if d < np.inf:
            candidates.append((d, cs_pos))
    candidates.sort()
    candidates = candidates[:config.cs_candidates_k]
    if not candidates:
        return None

    # Estimate average charge time for queue wait estimation
    avg_ct = config.compute_charge_time(
        config.battery_max * 0.3, config.battery_max * config.charge_target_threshold)
    gamma = compute_urgency(agent.soc)

    best_j, best_cs = float('inf'), None
    for d_cs, cs_pos in candidates:
        t_travel = d_cs * config.cost_move
        # Queue: number of agents already assigned × average charge time
        q_len = cs_queue.get_queue_length(cs_pos)
        t_wait = q_len * avg_ct
        # Battery at arrival
        b_arr = max(0.0, agent.battery - d_cs * config.energy_move)
        b_target = config.battery_max * config.charge_target_threshold
        t_charge = config.compute_charge_time(b_arr, b_target)
        t_comp = t_travel + t_wait + t_charge + config.charge_buffer_time

        risk = 0.0
        if b_arr < config.risk_threshold:
            risk = config.risk_coefficient * (config.risk_threshold - b_arr)

        j = (config.w_end * t_comp + config.w_travel * t_travel +
             (config.w_queue + gamma) * t_wait + risk)
        if j < best_j:
            best_j = j
            best_cs = cs_pos
    return best_cs


def select_cs_nearest(
    agent: Agent, grid_map: GridMap, config: SimConfig,
    bfs_cache: BFSCache,
) -> Optional[Tuple[int, int]]:
    """Nearest reachable CS (baseline) — spec §11.3.

    Reachable = BFS distance < ∞ and the agent can arrive with non-negative
    battery (b - d_cs * energy_move ≥ 0).
    """
    if not grid_map.cs_positions:
        return None
    dist_map = bfs_cache.get_distance_from(agent.x, agent.y)
    best_d, best_cs = float('inf'), None
    for cs_pos in grid_map.cs_positions:
        d = dist_map[cs_pos[1], cs_pos[0]]
        if d == np.inf:
            continue
        if agent.battery - d * config.energy_move < 0:
            continue
        if d < best_d:
            best_d = d
            best_cs = cs_pos
    return best_cs


def should_charge_before_task(
    agent: Agent, next_goal: Tuple[int, int],
    config: SimConfig, bfs_cache: BFSCache,
) -> bool:
    """Section 4.4.2 conditions 1–4."""
    gx, gy = next_goal
    dist = bfs_cache.get_distance_from(gx, gy)
    d = dist[agent.y, agent.x]
    if d == np.inf:
        return True
    b_at_goal = agent.battery - d * config.energy_move
    if b_at_goal < config.goal_min_battery:
        return True
    cs_dist = bfs_cache.get_cs_distance_map()
    d_goal_cs = cs_dist[gy, gx]
    if d_goal_cs == np.inf:
        return True
    if b_at_goal - d_goal_cs * config.energy_move < config.battery_safe_margin:
        return True
    if agent.soc < config.charge_start_threshold:
        return True
    return False


def assess_danger_level(agent: Agent, config: SimConfig) -> int:
    soc = agent.soc
    if soc <= config.emergency_l4_soc: return 4
    if soc <= config.emergency_l3_soc: return 3
    if len(agent.battery_history) >= 5:
        r = agent.battery_history[-5:]
        dec = (r[0] - r[-1]) / len(r)
        if dec > 0:
            ttl = agent.battery / dec
            if ttl < 20: return 3
            if ttl < 50: return 2
    if soc < 0.10: return 2
    if soc < 0.20: return 1
    return 0
