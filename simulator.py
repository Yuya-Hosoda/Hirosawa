"""
Main simulation loop — Fig. 4.1.
Tick-by-tick charging with proper CS occupancy tracking.
"""
from __future__ import annotations
import random
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict

from config import SimConfig
from grid_map import GridMap
from agent import Agent, Action, PathStep
from heuristic import EnergyAwareHeuristic
from low_level_search import low_level_search, ReservationTable
from scheduler import (
    CSQueue, BFSCache,
    select_cs_proposed, select_cs_nearest,
    should_charge_before_task, assess_danger_level,
)


@dataclass
class SimStats:
    tasks_completed: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_tasks_completed: int = 0
    collisions: int = 0
    energy_depletions: int = 0
    charge_events: int = 0
    emergency_interventions: int = 0
    replan_count: int = 0
    planning_times: List[float] = field(default_factory=list)
    _cs_q_sum: Dict[Tuple[int,int], float] = field(default_factory=lambda: defaultdict(float))
    _cs_q_n: int = 0

    def record_cs(self, cs_positions, cs_queue):
        self._cs_q_n += 1
        for p in cs_positions:
            self._cs_q_sum[p] += cs_queue.get_queue_length(p)

    def avg_cs_queue_length(self) -> float:
        if self._cs_q_n == 0 or not self._cs_q_sum:
            return 0.0
        return sum(self._cs_q_sum.values()) / (self._cs_q_n * len(self._cs_q_sum))


class Simulator:
    def __init__(self, grid_map: GridMap, config: SimConfig,
                 cs_strategy: str = 'proposed'):
        self.grid_map = grid_map
        self.config = config
        self.cs_strategy = cs_strategy
        self.agents: List[Agent] = []
        self.cs_queue = CSQueue()
        self.stats = SimStats()
        self.heuristic_cache: Dict[Tuple[int,int], EnergyAwareHeuristic] = {}
        self.bfs_cache = BFSCache(grid_map, config.agent_radius)
        self._charging: Dict[int, Tuple[float, Tuple[int,int]]] = {}  # aid -> (target_b, cs_pos)

    def add_agent(self, agent_id: int, x: int, y: int, z: int = 0,
                  battery: Optional[float] = None):
        b = battery if battery is not None else self.config.battery_max
        self.agents.append(Agent(agent_id=agent_id, x=x, y=y, z=z,
                                 battery=b, battery_max=self.config.battery_max))

    def generate_random_tasks(self, agent: Agent, n: int, rng: random.Random):
        r = self.config.agent_radius
        w, h = self.grid_map.width, self.grid_map.height
        for _ in range(n):
            for _ in range(200):
                gx, gy = rng.randint(r+1, w-r-2), rng.randint(r+1, h-r-2)
                if self.grid_map.can_agent_occupy(gx, gy, r):
                    agent.add_user_task((gx, gy))
                    break

    def _select_cs(self, agent: Agent) -> Optional[Tuple[int,int]]:
        if self.cs_strategy == 'proposed':
            return select_cs_proposed(agent, self.grid_map, self.cs_queue,
                                      self.config, self.bfs_cache)
        return select_cs_nearest(agent, self.grid_map, self.config, self.bfs_cache)

    def _assign_next_goal(self, agent: Agent):
        if agent.current_goal is not None:
            return
        task = agent.pop_next_user_task()
        if task is None:
            return
        if should_charge_before_task(agent, task, self.config, self.bfs_cache):
            cs = self._select_cs(agent)
            if cs is not None:
                agent.user_tasks.appendleft(task)
                agent.set_goal(cs, is_cs=True)
                self.cs_queue.assign_agent(cs, agent.agent_id)
                agent.charge_count += 1
                self.stats.charge_events += 1
                return
        agent.set_goal(task, is_cs=False)

    def _handle_emergency(self, agent: Agent):
        self.stats.emergency_interventions += 1
        if agent.current_goal is not None and not agent.current_goal_is_cs:
            agent.user_tasks.appendleft(agent.current_goal)
        cs = self._select_cs(agent)
        if cs is not None:
            agent.set_goal(cs, is_cs=True)
            self.cs_queue.assign_agent(cs, agent.agent_id)
            agent.charge_count += 1
            self.stats.charge_events += 1

    def _plan_paths(self):
        need = [a for a in self.agents
                if a.needs_replan and a.agent_id not in self._charging]
        if not need:
            return
        self.stats.replan_count += 1
        t0 = time.time()
        res = ReservationTable()
        for a in self.agents:
            if a.has_path and a not in need:
                pos = [(s.state.x, s.state.y, s.state.t) for s in a.planned_path[a.path_index:]]
                if pos: res.add_path(a.agent_id, pos)
        for agent in need:
            self._plan_single(agent, res)
            if agent.has_path:
                pos = [(s.state.x, s.state.y, s.state.t) for s in agent.planned_path[agent.path_index:]]
                if pos: res.add_path(agent.agent_id, pos)
        self.stats.planning_times.append(time.time() - t0)

    def _plan_single(self, agent: Agent, res: Optional[ReservationTable] = None):
        if agent.current_goal is None:
            return
        gx, gy = agent.current_goal
        key = (gx, gy)
        if key not in self.heuristic_cache:
            self.heuristic_cache[key] = EnergyAwareHeuristic(
                self.grid_map, gx, gy, self.config)
        path = low_level_search(
            self.grid_map, agent.agent_id,
            agent.x, agent.y, agent.z,
            agent.battery, 0, gx, gy,
            self.config, constraints=[], reservation_table=res,
            heuristic_obj=self.heuristic_cache[key])
        if path is not None:
            agent.planned_path = path
            agent.path_index = 0

    def _execute_step(self, agent: Agent):
        # Currently charging
        if agent.agent_id in self._charging:
            target_b, cs_pos = self._charging[agent.agent_id]
            # Check if another agent is charging here (capacity=1)
            current_charger = self.cs_queue.charging_at.get(cs_pos)
            if current_charger is not None and current_charger != agent.agent_id:
                # Wait — CS is occupied by another agent
                agent.total_wait_time += 1
                return
            # This agent charges
            self.cs_queue.set_charging(cs_pos, agent.agent_id)
            rate = self.config.charge_rate_at(agent.battery)
            agent.battery = min(self.config.battery_max, agent.battery + rate)
            if agent.battery >= target_b:
                del self._charging[agent.agent_id]
                self.cs_queue.unassign_agent(cs_pos, agent.agent_id)
                agent.clear_goal()
            return

        step = agent.get_next_action()
        if step is None:
            agent.battery -= self.config.energy_wait
            agent.total_wait_time += 1
            return

        action = step.action
        target = step.state
        if action == Action.CHARGE:
            rate = self.config.charge_rate_at(agent.battery)
            agent.battery = min(self.config.battery_max, agent.battery + rate)
        elif action in (Action.MOVE_RIGHT, Action.MOVE_LEFT,
                        Action.MOVE_DOWN, Action.MOVE_UP):
            agent.x, agent.y = target.x, target.y
            agent.battery -= self.config.energy_move
        elif action in (Action.ROTATE_UP, Action.ROTATE_DOWN):
            agent.z = target.z
            agent.battery -= self.config.energy_rotate
        elif action == Action.WAIT:
            agent.battery -= self.config.energy_wait

        agent.advance_path()
        if agent.battery < 0:
            self.stats.energy_depletions += 1
            agent.battery = 0

    def _check_completion(self, agent: Agent):
        if agent.agent_id in self._charging:
            return
        if agent.at_goal():
            if agent.current_goal_is_cs:
                cs_pos = agent.current_goal
                target_b = self.config.battery_max * self.config.charge_target_threshold
                if agent.battery < target_b:
                    # Check if CS is free
                    current = self.cs_queue.charging_at.get(cs_pos)
                    if current is not None and current != agent.agent_id:
                        # CS occupied — wait
                        agent.total_wait_time += 1
                        return
                    self._charging[agent.agent_id] = (target_b, cs_pos)
                    self.cs_queue.set_charging(cs_pos, agent.agent_id)
                else:
                    self.cs_queue.unassign_agent(cs_pos, agent.agent_id)
                    agent.clear_goal()
            else:
                agent.tasks_completed += 1
                self.stats.tasks_completed[agent.agent_id] += 1
                self.stats.total_tasks_completed += 1
                agent.clear_goal()

    def run(self, seed: int = 42, num_initial_tasks: int = 100,
            verbose: bool = False) -> SimStats:
        rng = random.Random(seed)
        self.stats = SimStats()
        self._charging.clear()
        self.cs_queue = CSQueue()

        for a in self.agents:
            a.battery = self.config.battery_max
            a.tasks_completed = 0
            a.charge_count = 0
            a.total_wait_time = 0
            a.battery_history = []
            a.planned_path = []
            a.path_index = 0
            a.current_goal = None
            a.current_goal_is_cs = False
            a.user_tasks.clear()
            a.execution_tasks.clear()
            self.generate_random_tasks(a, num_initial_tasks, rng)

        for a in self.agents:
            self._assign_next_goal(a)

        for t in range(self.config.max_time):
            # 1. Emergency
            for a in self.agents:
                if not a.current_goal_is_cs and a.agent_id not in self._charging:
                    if assess_danger_level(a, self.config) >= 3:
                        self._handle_emergency(a)

            # 2. Battery warning
            for a in self.agents:
                if (a.current_goal and not a.current_goal_is_cs
                        and a.agent_id not in self._charging
                        and a.soc < self.config.battery_warn_threshold and a.has_path):
                    rem = len(a.planned_path) - a.path_index
                    if a.battery < rem * self.config.energy_move + self.config.battery_safe_margin:
                        self._handle_emergency(a)

            # 3. Goal assignment
            for a in self.agents:
                if a.current_goal is None and a.agent_id not in self._charging:
                    self._assign_next_goal(a)
                    if a.current_goal is None and not a.user_tasks:
                        self.generate_random_tasks(a, 50, rng)
                        self._assign_next_goal(a)

            # 4. Planning
            self._plan_paths()

            # 5. Execute
            for a in self.agents:
                self._execute_step(a)

            # 6. Completion
            for a in self.agents:
                self._check_completion(a)

            # 7. Stats
            for a in self.agents:
                a.record_battery()
            self.stats.record_cs(self.grid_map.cs_positions, self.cs_queue)

            if verbose and t % 500 == 0:
                done = sum(a.tasks_completed for a in self.agents)
                avg_b = np.mean([a.battery for a in self.agents])
                nchg = len(self._charging)
                print(f"  t={t:4d} | tasks={done} | avg_b={avg_b:.0f} | "
                      f"charging={nchg} | depl={self.stats.energy_depletions}")

        return self.stats
