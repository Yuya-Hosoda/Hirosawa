"""
Configuration parameters for the Battery-Constrained MAPF system.
Based on: Hirosawa (2025), Table 5.2 and Chapter 4 equations.
"""
from dataclasses import dataclass


@dataclass
class SimConfig:
    # --- Simulation ---
    max_time: int = 3000
    random_seed: int = 42

    # --- Battery model (Table 5.2) ---
    battery_max: float = 2000.0
    energy_move: float = 5.0
    energy_wait: float = 0.0
    energy_rotate: float = 0.5

    # Piecewise-linear charging rates
    charge_rate_fast: float = 8.0     # SoC 0–40%
    charge_rate_medium: float = 4.0   # SoC 40–80%
    charge_rate_slow: float = 2.0     # SoC 80–100%
    charge_soc_boundary_low: float = 0.4
    charge_soc_boundary_high: float = 0.8

    charge_start_threshold: float = 0.30
    charge_target_threshold: float = 0.80

    # --- Time costs (Eq. 4.6–4.9) ---
    cost_move: float = 1.0
    cost_rotate: float = 1.0       # per z±1 step (full turn = 5 steps = 5 ticks)
    cost_wait: float = 1.0

    # --- Rotation layers ---
    num_rotation_layers: int = 6

    # --- SoC discretization ---
    soc_levels: int = 100

    # --- Agent ---
    agent_radius: int = 1

    # --- Low-Level Search ---
    ll_weight: float = 1.5
    ll_max_expansions: int = 20000
    ll_max_generated: int = 50000
    goal_min_battery: float = 10.0

    # --- High-Level Search (ECBS) ---
    hl_weight: float = 1.2
    hl_max_ct_nodes: int = 5000

    # --- Scheduler ---
    cs_candidates_k: int = 3
    battery_warn_threshold: float = 0.15
    battery_safe_margin: float = 50.0
    emergency_l4_soc: float = 0.02
    emergency_l3_soc: float = 0.04

    # CS selection weights (Eq. 4.57)
    w_end: float = 1.0
    w_travel: float = 0.5
    w_queue: float = 2.0
    risk_threshold: float = 100.0
    risk_coefficient: float = 5.0
    charge_buffer_time: float = 5.0

    @property
    def avg_charge_rate(self) -> float:
        return 0.4*self.charge_rate_fast + 0.4*self.charge_rate_medium + 0.2*self.charge_rate_slow

    @property
    def soc_delta(self) -> float:
        return self.battery_max / self.soc_levels

    def discretize_battery(self, b: float) -> float:
        delta = self.soc_delta
        return round(b / delta) * delta

    def battery_to_level(self, b: float) -> int:
        return int(round(b / self.soc_delta))

    def charge_rate_at(self, b: float) -> float:
        soc = b / self.battery_max
        if soc < self.charge_soc_boundary_low:
            return self.charge_rate_fast
        elif soc < self.charge_soc_boundary_high:
            return self.charge_rate_medium
        else:
            return self.charge_rate_slow

    def compute_charge_time(self, b_start: float, b_target: float) -> float:
        if b_target <= b_start:
            return 0.0
        b_target = min(b_target, self.battery_max)
        t = 0.0
        b = b_start
        bl = self.battery_max * self.charge_soc_boundary_low
        bh = self.battery_max * self.charge_soc_boundary_high
        if b < bl and b_target > b:
            se = min(b_target, bl); t += (se - b) / self.charge_rate_fast; b = se
        if b < bh and b_target > b:
            se = min(b_target, bh); t += (se - b) / self.charge_rate_medium; b = se
        if b < self.battery_max and b_target > b:
            se = min(b_target, self.battery_max); t += (se - b) / self.charge_rate_slow; b = se
        return t
