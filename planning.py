#!/usr/bin/env python3
"""
planning.py  -  AI Gym Planner
================================
Uses the `pddl` package to build a
STRIPS planning domain and problem modelling a strength-training macrocycle.
A built-in forward BFS planner resolves the action sequence, which is then
expanded into a full week-by-week lifting programme exported to lifting_logs.csv.

Usage:
    pip install pddl
    python planning.py
"""

import csv
from datetime import date, timedelta
from collections import deque

# Modify these values for a specific athlete before running.
USER = {
    "name":                   "Athlete",
    "current_squat_1rm":       100,   # kg  (current tested max)
    "current_bench_1rm":        80,   # kg
    "current_deadlift_1rm":    130,   # kg
    "target_squat_1rm":        140,   # kg  (end-of-cycle goal)
    "target_bench_1rm":        110,   # kg
    "target_deadlift_1rm":     180,   # kg
    "training_days_per_week":    4,
    "start_date":              date.today(),
}

def _build_pddl_with_package(user):
    """Use the pddl package (AI-Planning/pddl) to construct domain + problem."""
    from pddl.logic import Predicate
    from pddl.core import Domain, Problem
    from pddl.action import Action
    from pddl.requirements import Requirements

    reqs = [
        Requirements.STRIPS,
        Requirements.TYPING,
        Requirements.NEG_PRECONDITION,
    ]

    # Zero-arity predicates act as propositional state variables
    base_built   = Predicate("base-built")
    hyp_built    = Predicate("hypertrophy-built")
    goal_reached = Predicate("goal-reached")

    # Define actions
    do_base = Action(
        "do-base-block",
        parameters=[],
        precondition=~base_built(),
        effect=base_built(),
    )
    do_hyp = Action(
        "do-hypertrophy-block",
        parameters=[],
        precondition=base_built() & ~hyp_built(),
        effect=hyp_built(),
    )
    do_peak = Action(
        "do-peaking-block",
        parameters=[],
        precondition=hyp_built() & ~goal_reached(),
        effect=goal_reached(),
    )

    domain = Domain(
        "gym-planner",
        requirements=reqs,
        predicates=[base_built, hyp_built, goal_reached],
        actions=[do_base, do_hyp, do_peak],
    )

    # Derive initial state from proximity to the strength goal
    squat_ratio = user["current_squat_1rm"] / user["target_squat_1rm"]
    init = []
    if squat_ratio >= 0.75:
        init.append(base_built())    # athlete can skip base block
    if squat_ratio >= 0.90:
        init.append(hyp_built())     # athlete can skip hypertrophy block

    problem = Problem(
        "user-macrocycle",
        domain=domain,
        requirements=reqs,
        init=init,
        goal=goal_reached(),
    )

    return str(domain), str(problem), squat_ratio


def _build_pddl_strings(user):
    """Fallback: build PDDL strings manually without the pddl package."""
    squat_ratio = user["current_squat_1rm"] / user["target_squat_1rm"]
    init_atoms  = []
    if squat_ratio >= 0.75:
        init_atoms.append("(base-built)")
    if squat_ratio >= 0.90:
        init_atoms.append("(hypertrophy-built)")

    domain_str = (
        "(define (domain gym-planner)\n"
        "  (:requirements :strips :typing :negative-preconditions)\n"
        "  (:predicates\n"
        "    (base-built)\n"
        "    (hypertrophy-built)\n"
        "    (goal-reached))\n\n"
        "  (:action do-base-block\n"
        "    :parameters ()\n"
        "    :precondition (not (base-built))\n"
        "    :effect       (base-built))\n\n"
        "  (:action do-hypertrophy-block\n"
        "    :parameters ()\n"
        "    :precondition (and (base-built) (not (hypertrophy-built)))\n"
        "    :effect       (hypertrophy-built))\n\n"
        "  (:action do-peaking-block\n"
        "    :parameters ()\n"
        "    :precondition (and (hypertrophy-built) (not (goal-reached)))\n"
        "    :effect       (goal-reached))\n)"
    )

    init_line   = "    " + " ".join(init_atoms) if init_atoms else ""
    problem_str = (
        "(define (problem user-macrocycle)\n"
        "  (:domain gym-planner)\n"
        "  (:requirements :strips :typing :negative-preconditions)\n"
        f"  (:init\n{init_line})\n"
        "  (:goal (goal-reached))\n)"
    )
    return domain_str, problem_str, squat_ratio


def build_pddl_model(user):
    """Return (domain_str, problem_str, squat_ratio) using pddl package if available."""
    try:
        return _build_pddl_with_package(user)
    except ImportError:
        print("[WARNING] `pddl` package not found – using string fallback.")
        print("          Install with: pip install pddl")
        return _build_pddl_strings(user)


# minimal STRIPS forward-search planner.
_STRIPS_ACTIONS = [
    {
        "name":    "do-base-block",
        "pre":     frozenset(),
        "neg_pre": frozenset({"base-built"}),
        "add":     frozenset({"base-built"}),
        "del":     frozenset(),
    },
    {
        "name":    "do-hypertrophy-block",
        "pre":     frozenset({"base-built"}),
        "neg_pre": frozenset({"hypertrophy-built"}),
        "add":     frozenset({"hypertrophy-built"}),
        "del":     frozenset(),
    },
    {
        "name":    "do-peaking-block",
        "pre":     frozenset({"hypertrophy-built"}),
        "neg_pre": frozenset({"goal-reached"}),
        "add":     frozenset({"goal-reached"}),
        "del":     frozenset(),
    },
]


def bfs_planner(init_atoms: set, goal_atoms: set) -> list | None:
    """
    Forward BFS search over a ground STRIPS state space.

    Parameters
    ----------
    init_atoms : set of str   – ground atoms true in the initial state
    goal_atoms : set of str   – ground atoms that must be true in goal state

    Returns
    -------
    list of action-name strings (the plan), or None if unsolvable.
    """
    init_state = frozenset(init_atoms)
    goal_state = frozenset(goal_atoms)

    if goal_state <= init_state:
        return []   # already at goal

    queue   = deque([(init_state, [])])
    visited = {init_state}

    while queue:
        state, plan = queue.popleft()
        for action in _STRIPS_ACTIONS:
            if action["pre"] <= state and action["neg_pre"].isdisjoint(state):
                new_state = (state | action["add"]) - action["del"]
                new_plan  = plan + [action["name"]]
                if goal_state <= new_state:
                    return new_plan
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, new_plan))

    return None   # no plan exists



BLOCK_CONFIG: dict = {
    "do-base-block": {
        "label":          "Base / GPP",
        "weeks":          5,
        "intensity_low":  0.65,   # % of 1RM at start of block
        "intensity_high": 0.75,   # % of 1RM at end of block
        "sets":           4,
        "reps":           5,
        "rpe":            7,
        "weekly_gain":    0.025,  # estimated 1RM growth per non-deload week
        "desc": "Build a general strength foundation with moderate loads.",
    },
    "do-hypertrophy-block": {
        "label":          "Hypertrophy",
        "weeks":          6,
        "intensity_low":  0.70,
        "intensity_high": 0.80,
        "sets":           4,
        "reps":           8,
        "rpe":            8,
        "weekly_gain":    0.020,
        "desc": "Accumulate training volume to drive muscle growth.",
    },
    "do-peaking-block": {
        "label":          "Peaking",
        "weeks":          4,
        "intensity_low":  0.85,
        "intensity_high": 0.95,
        "sets":           3,
        "reps":           3,
        "rpe":            9,
        "weekly_gain":    0.030,
        "desc": "Maximize strength expression for testing or competition.",
    },
}

MAIN_LIFTS = ["Squat", "Bench Press", "Deadlift"]

ACCESSORIES: dict = {
    "Squat":       ["Romanian Deadlift", "Leg Press",       "Bulgarian Split Squat"],
    "Bench Press": ["Incline DB Press",  "Tricep Pushdown", "Cable Flye"],
    "Deadlift":    ["Barbell Row",       "Lat Pulldown",    "Good Morning"],
}

# 4-day upper/lower split
WEEKLY_SCHEDULE = [
    {"day": "Monday",   "main": "Squat",       "focus": "Lower"},
    {"day": "Tuesday",  "main": "Bench Press",  "focus": "Upper"},
    {"day": "Thursday", "main": "Deadlift",     "focus": "Lower"},
    {"day": "Friday",   "main": "Bench Press",  "focus": "Upper"},
]

_DAY_OFFSET: dict = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
}

CSV_FIELDS = [
    "Week", "Date", "Block", "Block_Week", "Is_Deload",
    "Day", "Focus", "Exercise", "Exercise_Type",
    "Sets", "Reps", "Weight_kg", "Intensity_Pct", "RPE_Target", "Est_1RM_kg",
]


def _round_weight(weight_kg: float, increment: float = 2.5) -> float:
    """Round to nearest bar-loadable increment (default 2.5 kg)."""
    return round(weight_kg / increment) * increment


# Loops over each block -> each week -> each session and builds a csv row per excerise

def generate_logs(plan: list, user: dict) -> list:
    """
    Expand an ordered list of PDDL action names into a full lifting log.

    Rules applied per block:
      * Intensity linearly ramps from intensity_low to intensity_high over the block.
      * A deload is inserted on week 4 of any block >= 4 weeks:
            intensity x0.85, sets -1, reps +2, RPE -1.
      * Estimated 1RM increases by weekly_gain each non-deload week.
      * Each session: one main compound lift + 3 accessories at 3x12 @ ~50% 1RM.

    Returns a list of row dicts matching CSV_FIELDS.
    """
    rows   = []
    week_n = 1
    one_rms = {
        "Squat":       user["current_squat_1rm"],
        "Bench Press": user["current_bench_1rm"],
        "Deadlift":    user["current_deadlift_1rm"],
    }

    # Anchor on the first Monday on or after start_date
    start      = user["start_date"]
    week_start = start + timedelta(days=(7 - start.weekday()) % 7)

    for action in plan:
        cfg = BLOCK_CONFIG[action]

        for bw in range(1, cfg["weeks"] + 1):
            deload = (bw % 4 == 0) and cfg["weeks"] >= 4

            # Intensity ramp (linear progression across the block)
            progress = (bw - 1) / max(cfg["weeks"] - 1, 1)
            pct      = cfg["intensity_low"] + progress * (
                cfg["intensity_high"] - cfg["intensity_low"]
            )
            if deload:
                pct *= 0.85

            sets = max(2, cfg["sets"] - (1 if deload else 0))
            reps = cfg["reps"] + (2 if deload else 0)
            rpe  = cfg["rpe"]  - (1 if deload else 0)

            for sess in WEEKLY_SCHEDULE:
                lift         = sess["main"]
                session_date = week_start + timedelta(days=_DAY_OFFSET[sess["day"]])
                main_wt      = _round_weight(one_rms[lift] * pct)

                # Main compound lift
                rows.append({
                    "Week":          week_n,
                    "Date":          session_date.isoformat(),
                    "Block":         cfg["label"],
                    "Block_Week":    bw,
                    "Is_Deload":     deload,
                    "Day":           sess["day"],
                    "Focus":         sess["focus"],
                    "Exercise":      lift,
                    "Exercise_Type": "Main",
                    "Sets":          sets,
                    "Reps":          reps,
                    "Weight_kg":     main_wt,
                    "Intensity_Pct": round(pct * 100, 1),
                    "RPE_Target":    rpe,
                    "Est_1RM_kg":    round(one_rms[lift], 1),
                })

                # Accessory work at ~50% of main lift 1RM
                for acc in ACCESSORIES[lift]:
                    rows.append({
                        "Week":          week_n,
                        "Date":          session_date.isoformat(),
                        "Block":         cfg["label"],
                        "Block_Week":    bw,
                        "Is_Deload":     deload,
                        "Day":           sess["day"],
                        "Focus":         sess["focus"],
                        "Exercise":      acc,
                        "Exercise_Type": "Accessory",
                        "Sets":          3,
                        "Reps":          12,
                        "Weight_kg":     _round_weight(one_rms[lift] * 0.50),
                        "Intensity_Pct": 50.0,
                        "RPE_Target":    7,
                        "Est_1RM_kg":    round(one_rms[lift], 1),
                    })

            # Progress estimated 1RMs after non-deload weeks
            if not deload:
                for lift in MAIN_LIFTS:
                    one_rms[lift] = round(one_rms[lift] * (1 + cfg["weekly_gain"]), 2)

            week_start += timedelta(weeks=1)
            week_n     += 1

    return rows


# Main function
def main():
    SEP = "=" * 64

    print(SEP)
    print("  AI GYM PLANNER  -  PDDL Macrocycle Generator")
    print(SEP)
    print(f'  Athlete  : {USER["name"]}')
    print(f'  Squat    : {USER["current_squat_1rm"]} kg  ->  target {USER["target_squat_1rm"]} kg')
    print(f'  Bench    : {USER["current_bench_1rm"]} kg  ->  target {USER["target_bench_1rm"]} kg')
    print(f'  Deadlift : {USER["current_deadlift_1rm"]} kg  ->  target {USER["target_deadlift_1rm"]} kg')
    print(SEP)

    # 1. Build PDDL model
    domain_str, problem_str, squat_ratio = build_pddl_model(USER)
    print("\n[PDDL DOMAIN]")
    print(domain_str)
    print("\n[PDDL PROBLEM]")
    print(problem_str)
    with open("gym_domain.pddl",  "w") as f: f.write(domain_str)
    with open("gym_problem.pddl", "w") as f: f.write(problem_str)
    print("\nSaved -> gym_domain.pddl, gym_problem.pddl")

    # 2. Determine initial state atoms from user profile
    init_atoms: set = set()
    if squat_ratio >= 0.75:
        init_atoms.add("base-built")
    if squat_ratio >= 0.90:
        init_atoms.add("hypertrophy-built")

    # 3. Solve with forward BFS
    plan = bfs_planner(init_atoms, goal_atoms={"goal-reached"})
    if plan is None:
        print("\n[ERROR] No valid plan found. Check user profile and action definitions.")
        return

    total_weeks = sum(BLOCK_CONFIG[a]["weeks"] for a in plan)
    print(f"\n[PLAN]  {len(plan)} block(s)  -  {total_weeks} weeks total")
    for i, action in enumerate(plan, 1):
        cfg = BLOCK_CONFIG[action]
        print(f'  Step {i}: {cfg["label"]:16s}  ({cfg["weeks"]} wks)  {cfg["desc"]}')

    # 4. Expand plan into lifting logs and export CSV
    logs        = generate_logs(plan, USER)
    output_file = "planning_lifting_logs.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(logs)

    print(f"\n[OUTPUT]  {len(logs):,} entries written -> {output_file}")
    print(SEP)


if __name__ == "__main__":
    main()