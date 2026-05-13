"""
ECBS (Enhanced Conflict-Based Search) — Section 4.3.
High-level search using Constraint Tree with FOCAL list.
"""
from __future__ import annotations
import heapq
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

from config import SimConfig
from grid_map import GridMap
from heuristic import EnergyAwareHeuristic
from low_level_search import (
    low_level_search, Constraint, VertexConstraint,
    EdgeConstraint, ReservationTable
)
from conflict_detection import detect_conflicts, Conflict
from agent import PathStep, Agent


@dataclass
class CTNode:
    """Constraint Tree node (Section 4.3.1)."""
    constraints: List[Constraint]
    paths: Dict[int, List[PathStep]]     # agent_id -> path
    cost: float                          # f(N) = sum of path costs (Eq. 4.49)
    num_conflicts: int = 0
    node_id: int = 0

    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        return self.num_conflicts < other.num_conflicts


def _path_cost(path: List[PathStep]) -> float:
    """Compute cost of a single agent's path."""
    if not path:
        return float('inf')
    return sum(step.cost for step in path)


def _node_cost(paths: Dict[int, List[PathStep]]) -> float:
    """Compute total cost f(N) = Σ cost(π_i) — Eq. 4.49."""
    return sum(_path_cost(p) for p in paths.values())


def _count_conflicts(paths: Dict[int, List[PathStep]],
                     grid_map: GridMap, config: SimConfig) -> int:
    """Count number of conflicts in current paths."""
    return len(detect_conflicts(paths, grid_map, config))


def ecbs_search(
    grid_map: GridMap,
    agents: List[Agent],
    config: SimConfig,
    existing_reservations: Optional[ReservationTable] = None,
    heuristic_cache: Optional[Dict[Tuple[int, int], EnergyAwareHeuristic]] = None,
) -> Optional[Dict[int, List[PathStep]]]:
    """Run ECBS to find collision-free paths for all agents.

    Args:
        grid_map: the grid environment
        agents: list of agents needing paths (must have current_goal set)
        config: simulation config
        existing_reservations: reservations from agents not being replanned
        heuristic_cache: cached heuristic objects keyed by (goal_x, goal_y)

    Returns:
        Dict mapping agent_id -> path, or None if failed.
    """
    if not agents:
        return {}

    if heuristic_cache is None:
        heuristic_cache = {}

    w_hl = config.hl_weight
    node_counter = 0

    # Build initial paths for all agents (root CT node)
    root_paths: Dict[int, List[PathStep]] = {}
    root_constraints: List[Constraint] = []

    # Build reservation table including existing reservations
    res_table = existing_reservations.clone() if existing_reservations else ReservationTable()

    for agent in agents:
        if agent.current_goal is None:
            continue

        gx, gy = agent.current_goal
        h_key = (gx, gy)
        if h_key not in heuristic_cache:
            heuristic_cache[h_key] = EnergyAwareHeuristic(
                grid_map, gx, gy, config
            )

        path = low_level_search(
            grid_map=grid_map,
            agent_id=agent.agent_id,
            start_x=agent.x, start_y=agent.y, start_z=agent.z,
            start_battery=agent.battery,
            start_time=0,
            goal_x=gx, goal_y=gy,
            config=config,
            constraints=[],
            reservation_table=res_table,
            heuristic_obj=heuristic_cache[h_key],
            min_goal_battery=(
                config.battery_max * config.charge_target_threshold
                if agent.current_goal_is_cs else config.goal_min_battery
            ),
        )

        if path is None:
            # Low-level search failure (Section 4.3.3)
            root_paths[agent.agent_id] = []
        else:
            root_paths[agent.agent_id] = path

    # Check if all agents have valid paths
    for agent in agents:
        if agent.current_goal is not None and not root_paths.get(agent.agent_id):
            # If any agent can't find a path at root, try without reservations
            gx, gy = agent.current_goal
            path = low_level_search(
                grid_map=grid_map,
                agent_id=agent.agent_id,
                start_x=agent.x, start_y=agent.y, start_z=agent.z,
                start_battery=agent.battery,
                start_time=0,
                goal_x=gx, goal_y=gy,
                config=config,
                constraints=[],
                reservation_table=None,
                heuristic_obj=heuristic_cache.get((gx, gy)),
                min_goal_battery=(
                    config.battery_max * config.charge_target_threshold
                    if agent.current_goal_is_cs else config.goal_min_battery
                ),
            )
            if path is not None:
                root_paths[agent.agent_id] = path

    root_cost = _node_cost(root_paths)
    root_conflicts = detect_conflicts(root_paths, grid_map, config)

    root_node = CTNode(
        constraints=root_constraints,
        paths=root_paths,
        cost=root_cost,
        num_conflicts=len(root_conflicts),
        node_id=node_counter,
    )
    node_counter += 1

    # If no conflicts, return immediately
    if len(root_conflicts) == 0:
        return root_paths

    # OPEN and FOCAL lists (Eq. 4.50-4.51)
    open_list: List[Tuple[float, int, CTNode]] = []
    heapq.heappush(open_list, (root_cost, root_node.node_id, root_node))

    ct_expansions = 0

    while open_list:
        ct_expansions += 1
        if ct_expansions > config.hl_max_ct_nodes:
            # Return best solution found so far
            break

        # Find f_min (Eq. 4.50)
        f_min = open_list[0][0]

        # Build FOCAL list (Eq. 4.51)
        focal_candidates = []
        for cost_val, nid, node in open_list:
            if cost_val <= w_hl * f_min:
                focal_candidates.append((node.num_conflicts, nid, node))

        if not focal_candidates:
            _, _, current = heapq.heappop(open_list)
        else:
            # Select from FOCAL based on secondary criterion (fewest conflicts)
            focal_candidates.sort()
            best_focal = focal_candidates[0][2]

            # Remove from open_list
            new_open = []
            found = False
            for item in open_list:
                if item[2].node_id == best_focal.node_id and not found:
                    found = True
                    continue
                new_open.append(item)
            open_list = new_open
            heapq.heapify(open_list)
            current = best_focal

        # Detect conflicts in current node
        conflicts = detect_conflicts(current.paths, grid_map, config)
        if len(conflicts) == 0:
            return current.paths

        # Select first conflict to resolve
        conflict = conflicts[0]

        # Generate child CT nodes (branch on the conflict)
        for constrained_agent in [conflict.agent_i, conflict.agent_j]:
            # Create an agent-specific constraint from the conflict.  For
            # distance-based vertex conflicts the two agents may be on different
            # cells, so each branch forbids that agent's own center position.
            if conflict.conflict_type in ('vertex', 'cs_capacity'):
                if constrained_agent == conflict.agent_i:
                    cx, cy = conflict.x, conflict.y
                else:
                    cx, cy = conflict.j_x, conflict.j_y
                new_constraint = VertexConstraint(
                    agent_id=constrained_agent, x=cx, y=cy, t=conflict.t
                )
            elif conflict.conflict_type == 'edge':
                if constrained_agent == conflict.agent_i:
                    x1, y1, x2, y2 = conflict.x, conflict.y, conflict.x2, conflict.y2
                else:
                    x1, y1, x2, y2 = conflict.j_x, conflict.j_y, conflict.j_x2, conflict.j_y2
                new_constraint = EdgeConstraint(
                    agent_id=constrained_agent, x1=x1, y1=y1, x2=x2, y2=y2, t=conflict.t
                )
            else:
                continue

            child_constraints = list(current.constraints) + [new_constraint]
            child_paths = dict(current.paths)

            # Replan for the constrained agent
            agent = None
            for a in agents:
                if a.agent_id == constrained_agent:
                    agent = a
                    break

            if agent is None or agent.current_goal is None:
                continue

            gx, gy = agent.current_goal
            h_key = (gx, gy)
            if h_key not in heuristic_cache:
                heuristic_cache[h_key] = EnergyAwareHeuristic(
                    grid_map, gx, gy, config
                )

            # Build reservation table from other agents' paths
            child_res = existing_reservations.clone() if existing_reservations else ReservationTable()

            new_path = low_level_search(
                grid_map=grid_map,
                agent_id=agent.agent_id,
                start_x=agent.x, start_y=agent.y, start_z=agent.z,
                start_battery=agent.battery,
                start_time=0,
                goal_x=gx, goal_y=gy,
                config=config,
                constraints=child_constraints,
                reservation_table=child_res,
                heuristic_obj=heuristic_cache[h_key],
                min_goal_battery=(
                    config.battery_max * config.charge_target_threshold
                    if agent.current_goal_is_cs else config.goal_min_battery
                ),
            )

            if new_path is None:
                # Low-level failure → f(N) = ∞ (Eq. 4.53)
                continue

            child_paths[constrained_agent] = new_path
            child_cost = _node_cost(child_paths)
            child_conflicts = detect_conflicts(child_paths, grid_map, config)

            child_node = CTNode(
                constraints=child_constraints,
                paths=child_paths,
                cost=child_cost,
                num_conflicts=len(child_conflicts),
                node_id=node_counter,
            )
            node_counter += 1

            heapq.heappush(open_list,
                           (child_cost, child_node.node_id, child_node))

    # Return best paths found (may have conflicts)
    if open_list:
        _, _, best = min(open_list, key=lambda x: (x[2].num_conflicts, x[0]))
        return best.paths

    return root_paths
