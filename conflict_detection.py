"""
Conflict detection for MoMo5's 3×3 occupancy model.
Implements vertex/edge/CS conflicts from Sections 4.1.1 and 4.3.2.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from agent import PathStep
from config import SimConfig
from grid_map import GridMap


@dataclass
class Conflict:
    """Detected conflict between two agents."""
    agent_i: int
    agent_j: int
    x: int
    y: int
    t: int
    conflict_type: str  # 'vertex', 'edge', 'cs_capacity'
    # For edge conflicts
    x2: int = 0
    y2: int = 0


def check_vertex_collision(x1: int, y1: int, x2: int, y2: int) -> bool:
    """Check vertex collision for 3×3 agents (Eq. 4.3).

    Two agents collide if:
        max(|Δx|, |Δy|) ≤ 2  AND  |Δx| + |Δy| ≤ 3
    """
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy) <= 2 and dx + dy <= 3


def detect_conflicts(
    paths: Dict[int, List[PathStep]],
    grid_map: GridMap,
    config: SimConfig
) -> List[Conflict]:
    """Detect all conflicts among a set of agent paths.

    Args:
        paths: dict mapping agent_id -> list of PathStep
        grid_map: the grid map
        config: simulation config

    Returns:
        List of detected Conflict objects.
    """
    conflicts = []
    agent_ids = sorted(paths.keys())

    # Build time-indexed position maps
    agent_positions: Dict[int, Dict[int, Tuple[int, int]]] = {}
    for aid in agent_ids:
        positions = {}
        for step in paths[aid]:
            positions[step.state.t] = (step.state.x, step.state.y)
        agent_positions[aid] = positions

    # Find time range
    all_times = set()
    for aid in agent_ids:
        all_times.update(agent_positions[aid].keys())
    if not all_times:
        return conflicts
    t_min = min(all_times)
    t_max = max(all_times)

    # Check pairwise conflicts
    for i_idx in range(len(agent_ids)):
        for j_idx in range(i_idx + 1, len(agent_ids)):
            ai = agent_ids[i_idx]
            aj = agent_ids[j_idx]
            pos_i = agent_positions[ai]
            pos_j = agent_positions[aj]

            for t in range(t_min, t_max + 1):
                # Get positions (if agent has no entry at time t,
                # use last known position)
                pi = _get_position_at(pos_i, t, t_max)
                pj = _get_position_at(pos_j, t, t_max)
                if pi is None or pj is None:
                    continue

                # Vertex conflict (Eq. 4.3)
                if check_vertex_collision(pi[0], pi[1], pj[0], pj[1]):
                    conflicts.append(Conflict(
                        agent_i=ai, agent_j=aj,
                        x=pi[0], y=pi[1], t=t,
                        conflict_type='vertex'
                    ))
                    break  # One conflict per pair per search

                # Edge conflict
                if t + 1 <= t_max:
                    pi_next = _get_position_at(pos_i, t + 1, t_max)
                    pj_next = _get_position_at(pos_j, t + 1, t_max)
                    if pi_next and pj_next:
                        # Check if agent_i at t+1 collides with agent_j at t
                        if check_vertex_collision(pi_next[0], pi_next[1],
                                                  pj[0], pj[1]):
                            if (pi[0], pi[1]) != (pi_next[0], pi_next[1]):
                                conflicts.append(Conflict(
                                    agent_i=ai, agent_j=aj,
                                    x=pi[0], y=pi[1], t=t,
                                    conflict_type='edge',
                                    x2=pi_next[0], y2=pi_next[1]
                                ))
                                break

    # CS capacity conflicts (Eq. 4.52)
    for t in range(t_min, t_max + 1):
        cs_occupants: Dict[Tuple[int, int], List[int]] = {}
        for aid in agent_ids:
            pos = _get_position_at(agent_positions[aid], t, t_max)
            if pos and grid_map.is_cs(pos[0], pos[1]):
                cs_key = (pos[0], pos[1])
                if cs_key not in cs_occupants:
                    cs_occupants[cs_key] = []
                cs_occupants[cs_key].append(aid)

        for cs_pos, occupants in cs_occupants.items():
            cap = grid_map.cs_capacity.get(cs_pos, 1)
            if len(occupants) > cap:
                # CS capacity conflict
                conflicts.append(Conflict(
                    agent_i=occupants[0],
                    agent_j=occupants[1],
                    x=cs_pos[0], y=cs_pos[1], t=t,
                    conflict_type='cs_capacity'
                ))

    return conflicts


def _get_position_at(
    positions: Dict[int, Tuple[int, int]], t: int, t_max: int
) -> Optional[Tuple[int, int]]:
    """Get agent position at time t. If no exact entry, use nearest prior."""
    if t in positions:
        return positions[t]
    # Find latest position before t
    latest_t = -1
    for pt in positions:
        if pt <= t and pt > latest_t:
            latest_t = pt
    if latest_t >= 0:
        return positions[latest_t]
    return None
