"""
Low-Level Search: Battery-Constrained Weighted A* — Algorithm 1 (Section 4.2).
Implements the single-agent path planner with 4D state space (x, y, z, b, t).
"""
from __future__ import annotations
import heapq
import math
from typing import List, Tuple, Optional, Set, Dict, NamedTuple
from dataclasses import dataclass

from config import SimConfig
from grid_map import GridMap
from heuristic import EnergyAwareHeuristic
from agent import AgentState, PathStep, Action


class SearchNode:
    """A* search node in the 4D state space."""
    __slots__ = ['x', 'y', 'z', 'b', 't', 'g', 'f', 'parent', 'action', 'counter']

    def __init__(self, x: int, y: int, z: int, b: float, t: int,
                 g: float, f: float, parent, action: Optional[Action],
                 counter: int):
        self.x = x
        self.y = y
        self.z = z
        self.b = b
        self.t = t
        self.g = g
        self.f = f
        self.parent = parent
        self.action = action
        self.counter = counter  # tie-breaking

    def __lt__(self, other):
        # f-value first, then prefer higher battery (Sec 4.2.3)
        if self.f != other.f:
            return self.f < other.f
        if self.b != other.b:
            return self.b > other.b
        return self.counter < other.counter

    def state_key(self, soc_delta: float) -> Tuple:
        """Discrete state key κ(s) = (y, x, z, t, ⌊b/ΔSoC⌋) — Eq. 4.48."""
        b_level = int(self.b / soc_delta) if soc_delta > 0 else int(self.b)
        return (self.y, self.x, self.z, self.t, b_level)


# Constraint types used by ECBS
@dataclass(frozen=True)
class VertexConstraint:
    """⟨agent_id, x, y, t⟩: agent cannot be at (x,y) at time t."""
    agent_id: int
    x: int
    y: int
    t: int


@dataclass(frozen=True)
class EdgeConstraint:
    """Agent cannot move from (x1,y1) to (x2,y2) between t and t+1."""
    agent_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    t: int


@dataclass(frozen=True)
class RangeConstraint:
    """Agent cannot be at (x,y) during time range [t_start, t_end]."""
    agent_id: int
    x: int
    y: int
    t_start: int
    t_end: int


Constraint = VertexConstraint | EdgeConstraint | RangeConstraint


def _is_vertex_conflict(x1: int, y1: int, x2: int, y2: int) -> bool:
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy) <= 2 and dx + dy <= 3


def _is_edge_conflict(
    x1_t: int, y1_t: int, x1_t1: int, y1_t1: int,
    x2_t: int, y2_t: int, x2_t1: int, y2_t1: int,
) -> bool:
    return (
        _is_vertex_conflict(x1_t, y1_t, x2_t1, y2_t1) or
        _is_vertex_conflict(x2_t, y2_t, x1_t1, y1_t1)
    )


class ConstraintTable:
    """Efficient lookup of constraints for a specific agent."""

    def __init__(self, agent_id: int, constraints: List[Constraint]):
        self.agent_id = agent_id
        self.vertex_constraints: Dict[Tuple[int, int, int], bool] = {}
        self.edge_constraints: Set[Tuple[int, int, int, int, int]] = set()
        self.range_constraints: List[RangeConstraint] = []

        for c in constraints:
            if isinstance(c, VertexConstraint) and c.agent_id == agent_id:
                self.vertex_constraints[(c.x, c.y, c.t)] = True
            elif isinstance(c, EdgeConstraint) and c.agent_id == agent_id:
                self.edge_constraints.add((c.x1, c.y1, c.x2, c.y2, c.t))
            elif isinstance(c, RangeConstraint) and c.agent_id == agent_id:
                self.range_constraints.append(c)

    def is_constrained_vertex(self, x: int, y: int, t: int) -> bool:
        if (x, y, t) in self.vertex_constraints:
            return True
        for rc in self.range_constraints:
            if rc.x == x and rc.y == y and rc.t_start <= t <= rc.t_end:
                return True
        return False

    def is_constrained_edge(self, x1: int, y1: int, x2: int, y2: int, t: int) -> bool:
        return (x1, y1, x2, y2, t) in self.edge_constraints


class ReservationTable:
    """Time-space reservations from other agents' planned paths."""

    def __init__(self):
        self.vertex_reservations: Dict[Tuple[int, int, int], int] = {}
        self.edge_reservations: Set[Tuple[int, int, int, int, int]] = set()
        # Time-indexed for fast lookup
        self._by_time: Dict[int, List[Tuple[int, int, int]]] = {}

    def clone(self) -> "ReservationTable":
        other = ReservationTable()
        other.vertex_reservations = dict(self.vertex_reservations)
        other.edge_reservations = set(self.edge_reservations)
        other._by_time = {t: list(entries) for t, entries in self._by_time.items()}
        return other

    def add_path(self, agent_id: int, path_positions: List[Tuple[int, int, int]]):
        """Add agent path as reservations. path_positions: list of (x, y, t)."""
        for x, y, t in path_positions:
            self.vertex_reservations[(x, y, t)] = agent_id
            if t not in self._by_time:
                self._by_time[t] = []
            self._by_time[t].append((x, y, agent_id))
        for i in range(len(path_positions) - 1):
            x1, y1, t1 = path_positions[i]
            x2, y2, t2 = path_positions[i + 1]
            if t2 == t1 + 1 and (x1, y1) != (x2, y2):
                self.edge_reservations.add((x1, y1, x2, y2, t1))

    def is_reserved_vertex(self, x: int, y: int, t: int,
                           agent_radius: int = 1) -> bool:
        """Check vertex collision with 3x3 agent model (Eq. 4.3)."""
        entries = self._by_time.get(t)
        if not entries:
            return False
        for rx, ry, aid in entries:
            if _is_vertex_conflict(x, y, rx, ry):
                return True
        return False

    def is_reserved_edge(self, x1: int, y1: int, x2: int, y2: int,
                         t: int, agent_radius: int = 1) -> bool:
        """Check edge collision against reserved same-tick transitions."""
        for ox1, oy1, ox2, oy2, ot in self.edge_reservations:
            if ot != t:
                continue
            if _is_edge_conflict(x1, y1, x2, y2, ox1, oy1, ox2, oy2):
                return True
        return False


def _get_strategic_charge_levels(b_current: float, config: SimConfig,
                                 e_need: float) -> List[float]:
    """Generate strategic battery levels for charging (Section 4.2.3).
    Three candidates: minimum needed, proactive threshold, full charge."""
    levels = []
    delta = config.soc_delta

    # Level 1: minimum to reach goal
    b_min = e_need
    if b_min > b_current and b_min <= config.battery_max:
        b_min_disc = math.ceil(b_min / delta) * delta
        b_min_disc = min(b_min_disc, config.battery_max)
        levels.append(b_min_disc)

    # Level 2: proactive threshold (charge_target_threshold)
    b_proactive = config.battery_max * config.charge_target_threshold
    if b_proactive > b_current and b_proactive not in levels:
        b_pro_disc = math.ceil(b_proactive / delta) * delta
        b_pro_disc = min(b_pro_disc, config.battery_max)
        if b_pro_disc not in levels:
            levels.append(b_pro_disc)

    # Level 3: full charge
    if config.battery_max > b_current and config.battery_max not in levels:
        levels.append(config.battery_max)

    return levels


def low_level_search(
    grid_map: GridMap,
    agent_id: int,
    start_x: int, start_y: int, start_z: int,
    start_battery: float, start_time: int,
    goal_x: int, goal_y: int,
    config: SimConfig,
    constraints: Optional[List[Constraint]] = None,
    reservation_table: Optional[ReservationTable] = None,
    heuristic_obj: Optional[EnergyAwareHeuristic] = None,
    min_goal_battery: Optional[float] = None,
    disable_occupancy_check: Optional[bool] = None,
) -> Optional[List[PathStep]]:
    """Battery-constrained Weighted A* search — Algorithm 1.

    Returns:
        List of PathStep from start to goal, or None if no path found.
    """
    if constraints is None:
        constraints = []
    goal_min_battery = config.goal_min_battery if min_goal_battery is None else min_goal_battery
    if disable_occupancy_check is None:
        disable_occupancy_check = config.disable_occupancy_check
    ct = ConstraintTable(agent_id, constraints)

    # Build heuristic if not provided
    if heuristic_obj is None or heuristic_obj.min_goal_battery != goal_min_battery:
        heuristic_obj = EnergyAwareHeuristic(
            grid_map, goal_x, goal_y, config, min_goal_battery=goal_min_battery
        )

    soc_delta = config.soc_delta
    w_ll = config.ll_weight
    L = config.num_rotation_layers

    # Counter for tie-breaking
    counter = 0

    # Open list (min-heap) and closed/dominance sets
    open_list: List[SearchNode] = []
    closed: Dict[Tuple, float] = {}   # state_key -> best g

    # Initialize with all possible starting layers (Algorithm 1, lines 4-11)
    # But we know the agent's current layer, so start with that
    for init_z in [start_z]:
        h_val = heuristic_obj.compute(start_x, start_y, start_battery)
        if h_val == float('inf'):
            continue
        f_val = 0.0 + w_ll * h_val
        node = SearchNode(start_x, start_y, init_z, start_battery,
                          start_time, 0.0, f_val, None, None, counter)
        counter += 1

        key = node.state_key(soc_delta)
        if key not in closed or 0.0 < closed[key]:
            heapq.heappush(open_list, node)
            closed[key] = 0.0

    expansions = 0
    generated = 0

    while open_list:
        node = heapq.heappop(open_list)
        expansions += 1

        if expansions > config.ll_max_expansions:
            return None  # Cutoff

        # Goal check (Algorithm 1, line 14-15)
        if (node.x == goal_x and node.y == goal_y and
                node.b >= goal_min_battery):
            return _reconstruct_path(node)

        # Closed check (line 17)
        key = node.state_key(soc_delta)
        if key in closed and closed[key] < node.g:
            continue

        # --- Expand successors ---

        # Actions: Move, Rotate, Wait (line 18)
        successors = []

        # Move actions (Eq. 4.17-4.20)
        move_dirs = _get_move_directions(node.z, L)
        for dx, dy, action in move_dirs:
            nx, ny = node.x + dx, node.y + dy
            if disable_occupancy_check:
                if not grid_map.is_free(nx, ny):
                    continue
            elif not grid_map.can_agent_occupy(nx, ny, config.agent_radius):
                continue
            new_b = node.b - config.energy_move
            if new_b < 0:
                continue
            new_t = node.t + int(config.cost_move)
            g_new = node.g + config.cost_move

            # Constraint and reservation checks for the full action interval.
            if (ct.is_constrained_vertex(node.x, node.y, node.t) or
                    ct.is_constrained_vertex(nx, ny, new_t) or
                    ct.is_constrained_edge(node.x, node.y, nx, ny, node.t)):
                continue

            if reservation_table:
                if (reservation_table.is_reserved_vertex(node.x, node.y, node.t, config.agent_radius) or
                        reservation_table.is_reserved_vertex(nx, ny, new_t, config.agent_radius) or
                        reservation_table.is_reserved_edge(node.x, node.y, nx, ny, node.t, config.agent_radius)):
                    continue

            successors.append((nx, ny, node.z, new_b, new_t, g_new, action))

        # Rotate actions (Eq. 4.21-4.24)
        for dz, action in [(1, Action.ROTATE_UP), (-1, Action.ROTATE_DOWN)]:
            new_z = node.z + dz
            if new_z < 0 or new_z >= L:
                continue
            new_b = node.b - config.energy_rotate
            if new_b < 0:
                continue
            new_t = node.t + int(config.cost_rotate)
            g_new = node.g + config.cost_rotate

            if ct.is_constrained_vertex(node.x, node.y, new_t):
                continue

            if reservation_table:
                if reservation_table.is_reserved_vertex(node.x, node.y, new_t, config.agent_radius):
                    continue

            successors.append((node.x, node.y, new_z, new_b, new_t, g_new, action))

        # Wait action (Eq. 4.25-4.28)
        new_b_wait = node.b - config.energy_wait
        if new_b_wait >= 0:
            new_t_wait = node.t + int(config.cost_wait)
            g_wait = node.g + config.cost_wait
            if not ct.is_constrained_vertex(node.x, node.y, new_t_wait):
                skip = False
                if reservation_table:
                    if reservation_table.is_reserved_vertex(node.x, node.y, new_t_wait, config.agent_radius):
                        skip = True
                if not skip:
                    successors.append((node.x, node.y, node.z, new_b_wait,
                                       new_t_wait, g_wait, Action.WAIT))

        # Process Move/Rotate/Wait successors
        for sx, sy, sz, sb, st, sg, sa in successors:
            generated += 1
            if generated > config.ll_max_generated:
                return None  # Cutoff

            h_val = heuristic_obj.compute(sx, sy, sb)
            if h_val == float('inf'):
                continue
            f_val = sg + w_ll * h_val

            child = SearchNode(sx, sy, sz, sb, st, sg, f_val,
                               node, sa, counter)
            counter += 1

            ckey = child.state_key(soc_delta)
            if ckey in closed and closed[ckey] <= sg:
                continue
            closed[ckey] = sg
            heapq.heappush(open_list, child)

        # Charge actions at CS (Algorithm 1, lines 27-34)
        if (grid_map.is_cs(node.x, node.y) and
                node.b < config.battery_max):
            # Compute energy needed to reach goal from current position
            d_goal = heuristic_obj.dist_to_goal[node.y, node.x]
            e_need = d_goal * config.energy_move + goal_min_battery if d_goal < float('inf') else config.battery_max

            charge_levels = _get_strategic_charge_levels(node.b, config, e_need)

            for target_b in charge_levels:
                if target_b <= node.b:
                    continue

                # Compute charging time
                dt_charge = config.compute_charge_time(node.b, target_b)
                dt_charge_ticks = max(1, int(math.ceil(dt_charge)))

                new_b_ch = config.compute_battery_after_charge(node.b, dt_charge_ticks)
                new_t_ch = node.t + dt_charge_ticks
                g_ch = node.g + dt_charge_ticks

                # Check constraints for entire charging duration
                constrained = False
                for tc in range(node.t + 1, new_t_ch + 1):
                    if ct.is_constrained_vertex(node.x, node.y, tc):
                        constrained = True
                        break
                if constrained:
                    continue

                if reservation_table:
                    reserved = False
                    for tc in range(node.t + 1, new_t_ch + 1):
                        if reservation_table.is_reserved_vertex(
                                node.x, node.y, tc, config.agent_radius):
                            reserved = True
                            break
                    if reserved:
                        continue

                generated += 1
                if generated > config.ll_max_generated:
                    return None

                h_ch = heuristic_obj.compute(node.x, node.y, new_b_ch)
                if h_ch == float('inf'):
                    continue
                f_ch = g_ch + w_ll * h_ch

                child_ch = SearchNode(node.x, node.y, node.z, new_b_ch,
                                      new_t_ch, g_ch, f_ch, node,
                                      Action.CHARGE, counter)
                counter += 1

                ckey_ch = child_ch.state_key(soc_delta)
                if ckey_ch in closed and closed[ckey_ch] <= g_ch:
                    continue
                closed[ckey_ch] = g_ch
                heapq.heappush(open_list, child_ch)

    return None  # No solution found


import math


def _get_move_directions(z: int, L: int) -> List[Tuple[int, int, Action]]:
    """Get allowed move directions based on current rotation layer.

    z=0: horizontal movement only (left/right)
    z=L-1(=5): vertical movement only (up/down)
    z=1..4: intermediate layers — no movement allowed
    """
    if z == 0:
        # Horizontal movement
        return [(1, 0, Action.MOVE_RIGHT), (-1, 0, Action.MOVE_LEFT)]
    elif z == L - 1:
        # Vertical movement
        return [(0, 1, Action.MOVE_DOWN), (0, -1, Action.MOVE_UP)]
    else:
        # Intermediate rotation layers — no movement
        return []


def _reconstruct_path(node: SearchNode) -> List[PathStep]:
    """Reconstruct path from goal node to start."""
    path = []
    current = node
    while current.parent is not None:
        step = PathStep(
            state=AgentState(current.x, current.y, current.z,
                             current.b, current.t),
            action=current.action,
            cost=current.g - current.parent.g
        )
        path.append(step)
        current = current.parent
    # Add start state
    path.append(PathStep(
        state=AgentState(current.x, current.y, current.z,
                         current.b, current.t),
        action=Action.WAIT,  # placeholder for start
        cost=0.0
    ))
    path.reverse()
    return path
