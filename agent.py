"""
Agent model for MoMo5.
State representation s = (p, z, b, t) as defined in Eq. 4.5.
Task queue management for Lifelong MAPF (Section 4.1.3).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, NamedTuple
from collections import deque
from enum import IntEnum


class Action(IntEnum):
    """Agent actions (Section 4.2.1)."""
    MOVE_RIGHT = 0
    MOVE_LEFT = 1
    MOVE_DOWN = 2
    MOVE_UP = 3
    ROTATE_UP = 4    # z + 1
    ROTATE_DOWN = 5  # z - 1
    WAIT = 6
    CHARGE = 7


class AgentState(NamedTuple):
    """4D state s = (x, y, z, b, t) — Eq. 4.5."""
    x: int
    y: int
    z: int        # rotation layer 0..L-1
    b: float      # battery
    t: int        # time step


@dataclass
class PathStep:
    """One step in a planned path."""
    state: AgentState
    action: Action
    cost: float     # time cost of this action


@dataclass
class Agent:
    """MoMo5 agent with state, task queue, and planned path."""
    agent_id: int
    x: int
    y: int
    z: int                     # current rotation layer
    battery: float             # current battery level
    battery_max: float

    # Task management (Section 4.1.3)
    user_tasks: deque = field(default_factory=deque)     # W_i: user task queue
    execution_tasks: deque = field(default_factory=deque) # T_i: execution task queue
    current_goal: Optional[Tuple[int, int]] = None
    current_goal_is_cs: bool = False

    # Planned path
    planned_path: List[PathStep] = field(default_factory=list)
    path_index: int = 0

    # Statistics
    tasks_completed: int = 0
    charge_count: int = 0
    total_wait_time: int = 0
    battery_history: List[float] = field(default_factory=list)

    @property
    def state(self) -> AgentState:
        return AgentState(self.x, self.y, self.z, self.battery, 0)

    @property
    def soc(self) -> float:
        """State of Charge as fraction [0,1]."""
        return self.battery / self.battery_max

    @property
    def has_path(self) -> bool:
        return len(self.planned_path) > 0 and self.path_index < len(self.planned_path)

    @property
    def needs_replan(self) -> bool:
        return self.current_goal is not None and not self.has_path

    def add_user_task(self, goal: Tuple[int, int]):
        """Add a user-specified task w_{i,k} to queue W_i."""
        self.user_tasks.append(goal)

    def pop_next_user_task(self) -> Optional[Tuple[int, int]]:
        """Pop next task from W_i."""
        if self.user_tasks:
            return self.user_tasks.popleft()
        return None

    def set_goal(self, goal: Tuple[int, int], is_cs: bool = False):
        """Set current goal g_{i,m}."""
        self.current_goal = goal
        self.current_goal_is_cs = is_cs
        self.planned_path = []
        self.path_index = 0

    def clear_goal(self):
        self.current_goal = None
        self.current_goal_is_cs = False
        self.planned_path = []
        self.path_index = 0

    def at_goal(self) -> bool:
        if self.current_goal is None:
            return False
        return (self.x, self.y) == self.current_goal

    def record_battery(self):
        self.battery_history.append(self.battery)

    def get_next_action(self) -> Optional[PathStep]:
        """Get next planned action."""
        if self.has_path:
            return self.planned_path[self.path_index]
        return None

    def advance_path(self):
        """Move to next step in planned path."""
        self.path_index += 1
