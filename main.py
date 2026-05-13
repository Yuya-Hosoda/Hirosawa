"""
Main entry point — reproduces all experiments from Chapter 5.

Experiment 1: Safety filter verification (Section 5.2)
Experiment 2: SoC discretization granularity (Section 5.3)
Experiment 3: CS selection strategy comparison (Section 5.4)
"""
from __future__ import annotations
import sys
import time
import argparse
import numpy as np

from config import SimConfig
from grid_map import GridMap, create_scenario1, create_empty_map
from agent import Agent, Action
from heuristic import EnergyAwareHeuristic
from low_level_search import low_level_search
from simulator import Simulator


# ============================================================
#  Experiment 1: Safety filter (Section 5.2, Table 5.3)
# ============================================================
def run_experiment1():
    print("=" * 72)
    print("EXPERIMENT 1: Safety Filter Verification (Section 5.2)")
    print("=" * 72)

    # Spec §15.1: all cases use (start_x, start_y) = (0, 0).  Cases 1–6 and 8
    # use a 50×1 linear map; Case 7 uses a 20×20 2D map so the L-shaped
    # trajectory can include a rotation.  Occupancy check is disabled per
    # spec (single-agent safety-filter test).
    #
    # Tuple layout: (name, width, height, cs_list, goal, init_b, B_min,
    #                max_exp, max_gen)
    cases = [
        ("Unreachable (no CS)",       50, 1, [],        (49, 0), 200.0, 10.0,  5000, 10000),
        ("Boundary reachable (d=35)", 50, 1, [],        (35, 0), 190.0, 15.0,  5000, 10000),
        ("Boundary reachable (d=40)", 50, 1, [],        (40, 0), 210.0, 10.0,  5000, 10000),
        ("Unreachable (d=45)",        50, 1, [],        (45, 0), 210.0, 10.0,  5000, 10000),
        ("Charge required (via CS)",  50, 1, [(20, 0)], (45, 0), 150.0, 10.0,  5000, 10000),
        ("Goal is CS",                50, 1, [(20, 0)], (20, 0), 110.0, 10.0,  5000, 10000),
        ("L-shape with rotation",     20, 20, [],       (15, 15), 500.0, 30.0, 5000, 10000),
        ("Cutoff (tiny budget)",      50, 1, [(20, 0)], (45, 0), 150.0, 10.0,  5, 19),
    ]

    header = (f"{'Case':<35s} {'Status':<8s} {'S1':<4s} {'S2':<4s} "
              f"{'b_goal':>8s} {'b_min':>8s} {'Chrg':>5s} {'Rot':>5s} {'ms':>7s}")
    print(f"\n{header}")
    print("-" * len(header))

    for name, w, h, cs_list, goal, init_b, b_min, max_exp, max_gen in cases:
        cfg = SimConfig(disable_occupancy_check=True, goal_min_battery=b_min)
        cfg.ll_max_expansions = max_exp
        cfg.ll_max_generated = max_gen

        gm = create_empty_map(w, h, cs_list)

        t0 = time.time()
        path = low_level_search(
            grid_map=gm, agent_id=0,
            start_x=0, start_y=0, start_z=0,
            start_battery=init_b, start_time=0,
            goal_x=goal[0], goal_y=goal[1], config=cfg,
            min_goal_battery=b_min,
        )
        ms = (time.time() - t0) * 1000

        if path is None:
            # Cutoff iff the tiny-budget case was used; otherwise no solution.
            status = "Cutoff" if max_exp <= 10 else "No sol"
            print(f"{name:<35s} {status:<8s} {'--':<4s} {'--':<4s} "
                  f"{'--':>8s} {'--':>8s} {'--':>5s} {'--':>5s} {ms:>6.1f}")
        else:
            b_final = path[-1].state.b
            b_min_path = min(s.state.b for s in path)
            s1 = "OK" if b_min_path >= 0 else "FAIL"
            s2 = "OK" if b_final >= b_min else "FAIL"
            chrg = sum(1 for s in path if s.action is not None and s.action.value == 7)
            rot = sum(1 for s in path if s.action is not None and s.action.value in (4,5))
            print(f"{name:<35s} {'Found':<8s} {s1:<4s} {s2:<4s} "
                  f"{b_final:>8.2f} {b_min:>8.2f} {chrg:>5d} {rot:>5d} {ms:>6.1f}")
    print()


# ============================================================
#  Experiment 2: SoC discretization (Section 5.3)
# ============================================================
def run_experiment2():
    print("=" * 72)
    print("EXPERIMENT 2: SoC Discretization Granularity (Section 5.3)")
    print("=" * 72)

    soc_levels = [10, 20, 40, 100]

    # Spec §15.2 — Case A reuses Exp1 Case 2 settings (d=35, b=190, B_min=15);
    # Case B reuses Exp1 Case 5 settings (d=45, CS at (20,0), b=150, B_min=10).
    print("\n--- Case A: Boundary condition (d=35, b=190, B_min=15) ---")
    print(f"{'L_SoC':<8s} {'Status':<8s} {'Safety':<8s} {'ms':>8s}")
    print("-" * 36)
    for lsoc in soc_levels:
        cfg = SimConfig(soc_levels=lsoc, ll_max_expansions=5000, ll_max_generated=10000,
                        disable_occupancy_check=True, goal_min_battery=15.0)
        gm = create_empty_map(50, 1, [])
        t0 = time.time()
        path = low_level_search(gm, 0, 0, 0, 0, 190.0, 0, 35, 0, cfg,
                                min_goal_battery=15.0)
        ms = (time.time() - t0) * 1000
        if path is None:
            print(f"{lsoc:<8d} {'No sol':<8s} {'--':<8s} {ms:>7.1f}")
        else:
            ok = "OK" if path[-1].state.b >= 15.0 else "FAIL"
            print(f"{lsoc:<8d} {'Found':<8s} {ok:<8s} {ms:>7.1f}")

    print("\n--- Case B: Charge required (d=45, CS at (20,0), b=150, B_min=10) ---")
    print(f"{'L_SoC':<8s} {'Status':<8s} {'Safety':<8s} {'Chrg':>6s} {'PathLen':>8s} {'ms':>8s}")
    print("-" * 56)
    # Spec Table 5.5b: L_SoC=10 → 50000/50000, 20/40 → 100000/100000, 100 → 5000/10000.
    limits = {10: (50000, 50000), 20: (100000, 100000),
              40: (100000, 100000), 100: (5000, 10000)}
    for lsoc in soc_levels:
        max_exp, max_gen = limits[lsoc]
        cfg = SimConfig(soc_levels=lsoc, ll_max_expansions=max_exp,
                        ll_max_generated=max_gen,
                        disable_occupancy_check=True, goal_min_battery=10.0)
        gm = create_empty_map(50, 1, [(20, 0)])
        t0 = time.time()
        path = low_level_search(gm, 0, 0, 0, 0, 150.0, 0, 45, 0, cfg,
                                min_goal_battery=10.0)
        ms = (time.time() - t0) * 1000
        if path is None:
            print(f"{lsoc:<8d} {'No sol':<8s} {'--':<8s} {'--':>6s} {'--':>8s} {ms:>7.1f}")
        else:
            ok = "OK" if path[-1].state.b >= 10.0 else "FAIL"
            chrg = sum(1 for s in path if s.action is not None and s.action.value == 7)
            print(f"{lsoc:<8d} {'Found':<8s} {ok:<8s} {chrg:>6d} {len(path):>8d} {ms:>7.1f}")
    print()


# ============================================================
#  Experiment 3: CS selection strategy (Section 5.4)
# ============================================================
def run_experiment3():
    print("=" * 72)
    print("EXPERIMENT 3: CS Selection Strategy Comparison (Section 5.4)")
    print("=" * 72)

    agent_counts = [2, 3]
    strategies = ['proposed', 'nearest']
    # Spec §15.3: 20 trials per condition (seeds 0..19).
    num_seeds = 20

    results = {}

    for n_agents in agent_counts:
        for strategy in strategies:
            key = (n_agents, strategy)
            completed_list = []
            queue_lengths = []

            print(f"\n  Running: {n_agents} agents, {strategy}, {num_seeds} seeds...")
            t_batch = time.time()

            for seed in range(num_seeds):
                config = SimConfig()
                config.max_time = 3000
                gm = create_scenario1()
                sim = Simulator(gm, config, cs_strategy=strategy)

                starts = _get_starts(n_agents, gm, config)
                for i, (ax, ay) in enumerate(starts):
                    sim.add_agent(i, ax, ay, z=0, battery=config.battery_max)

                stats = sim.run(seed=seed, num_initial_tasks=200, verbose=False)
                completed_list.append(stats.total_tasks_completed)
                queue_lengths.append(stats.avg_cs_queue_length())

            elapsed = time.time() - t_batch
            mean_c = np.mean(completed_list)
            print(f"    Done in {elapsed:.1f}s — mean tasks: {mean_c:.1f}")
            results[key] = {'completed': completed_list, 'avg_queue': queue_lengths}

    # === Print Table 5.7 ===
    print(f"\n{'='*72}")
    print("Table 5.7: CS Selection Strategy Comparison")
    print(f"{'='*72}")
    print(f"{'Agents':<8s} {'Strategy':<12s} {'Mean':>8s} {'Std':>8s} "
          f"{'Min':>6s} {'Max':>6s} {'Improve':>10s}")
    print("-" * 60)

    for n_agents in agent_counts:
        p_data = results[(n_agents, 'proposed')]['completed']
        n_data = results[(n_agents, 'nearest')]['completed']
        mean_p, mean_n = np.mean(p_data), np.mean(n_data)
        imp = (mean_p - mean_n) / mean_n * 100 if mean_n > 0 else 0

        for strat in strategies:
            d = results[(n_agents, strat)]['completed']
            imp_str = f"+{imp:.1f}%" if strat == 'proposed' else ""
            print(f"{n_agents:<8d} {strat:<12s} {np.mean(d):>8.1f} {np.std(d):>8.1f} "
                  f"{min(d):>6d} {max(d):>6d} {imp_str:>10s}")

    # === Print Table 5.8 ===
    print(f"\n{'='*72}")
    print("Table 5.8: Average CS Queue Length")
    print(f"{'='*72}")
    print(f"{'Agents':<8s} {'proposed':>12s} {'nearest':>12s}")
    print("-" * 36)
    for n_agents in agent_counts:
        q_p = np.mean(results[(n_agents, 'proposed')]['avg_queue'])
        q_n = np.mean(results[(n_agents, 'nearest')]['avg_queue'])
        print(f"{n_agents:<8d} {q_p:>12.2f} {q_n:>12.2f}")
    print()


def _get_starts(n, gm, config):
    """Initial agent positions per spec §2.4.

    Spec coordinates are (y, x); GridMap uses (x, y), so positions are
    transposed below.  Both 2- and 3-agent layouts are taken from
    SCENARIO1_INITIAL_POSITIONS_{2,3}_AGENTS.
    """
    if n <= 2:
        # Spec: (4, 4), (4, 25) in (y, x)  →  (x=4, y=4), (x=25, y=4)
        candidates = [(4, 4), (25, 4)]
    else:
        # Spec: (4, 4), (4, 25), (25, 4) in (y, x)
        candidates = [(4, 4), (25, 4), (4, 25)]
    starts = []
    for cx, cy in candidates:
        if len(starts) >= n:
            break
        if gm.can_agent_occupy(cx, cy, config.agent_radius):
            starts.append((cx, cy))
    return starts[:n]


def main():
    parser = argparse.ArgumentParser(
        description="Battery-Constrained MAPF for MoMo5 — Experiment Reproduction")
    parser.add_argument('--exp', type=int, default=0, help='1/2/3 or 0=all')
    args = parser.parse_args()

    print("=" * 72)
    print("  Battery-Constrained Multi-Agent Path Finding for MoMo5")
    print("  Reproduction of: Hirosawa (2025) Bachelor's Thesis")
    print("=" * 72)

    if args.exp == 0 or args.exp == 1:
        run_experiment1()
    if args.exp == 0 or args.exp == 2:
        run_experiment2()
    if args.exp == 0 or args.exp == 3:
        run_experiment3()

    print("All requested experiments completed.")

if __name__ == '__main__':
    main()
