"""
Task 3: runs Tests A, B and C (plus extra cases) and records, for each:
initial state, goal, whether a plan was found, the plan, and whether the
plan is actually valid (checked independently with validate_plan).

Run:
    python run_tests.py            (output saved in test_results.txt)
"""

from planner import (INITIAL, GOAL, warehouse_actions, make_action, move,
                     pickup, drop, bfs_plan, validate_plan, fmt_state,
                     solve_and_report)


def irrelevant_actions():
    """Actions that change the robot (or nothing useful) but never the package."""
    return [
        # Recharge: affects only the robot's battery
        make_action("Recharge(A)", pos_pre=["At(Robot,A)"],
                    pos_eff=["Charged(Robot)"]),
        # Scan: records that a location was inspected
        make_action("Scan(C)", pos_pre=["At(Robot,C)"], pos_eff=["Scanned(C)"]),
        # Shortcut that moves ONLY the robot, and only when it is not
        # carrying the package (uses a NEGATIVE precondition).
        make_action("Shuttle(A,C)", pos_pre=["At(Robot,A)"],
                    neg_pre=["Holding(Package)"],
                    pos_eff=["At(Robot,C)"], neg_eff=["At(Robot,A)"]),
    ]


CASES = [
    ("Test A - solvable: original warehouse problem",
     INITIAL, GOAL, warehouse_actions()),

    ("Test B - impossible: PickUp actions removed",
     INITIAL, GOAL, warehouse_actions(include_pickup=False)),

    ("Test B2 - impossible: link B-C removed (C unreachable)",
     INITIAL, GOAL,
     [move("A", "B"), move("B", "A")] + [pickup(l) for l in "ABC"]
     + [drop(l) for l in "ABC"]),

    ("Test C - irrelevant actions added (robot-only moves, Recharge, Scan)",
     INITIAL, GOAL, warehouse_actions() + irrelevant_actions()),

    ("Test C2 - robot already at C, package still at A",
     frozenset({"At(Robot,C)", "At(Package,A)"}), GOAL, warehouse_actions()),
]


def check_handwritten_plan(title, plan):
    print("=" * 72)
    print(title)
    print("=" * 72)
    ok, lines = validate_plan(INITIAL, GOAL, plan)
    for line in lines:
        print("  " + line)
    print("Plan valid:", ok, "\n")


if __name__ == "__main__":
    summary = []
    for title, init, goal, acts in CASES:
        plan = solve_and_report(title, init, goal, acts)
        valid = validate_plan(init, goal, plan)[0] if plan is not None else None
        summary.append((title.split(" - ")[0], fmt_state(init), fmt_state(goal),
                        "yes" if plan is not None else "no (No plan found)",
                        ", ".join(a.name for a in plan) if plan else "-",
                        {True: "yes", False: "NO", None: "n/a"}[valid]))

    # The example sequence mentioned in the lab sheet (Task 1) - is it valid?
    check_handwritten_plan(
        "Check: example sequence from the sheet  Move(A,B), PickUp(Package,B), "
        "Move(B,C), Drop(Package,C)",
        [move("A", "B"), pickup("B"), move("B", "C"), drop("C")])

    # The hand-constructed plan from Task 1
    check_handwritten_plan(
        "Check: hand-constructed plan (Task 1)",
        [pickup("A"), move("A", "B"), move("B", "C"), drop("C")])

    # A plan that only moves the robot - must NOT satisfy the goal
    check_handwritten_plan(
        "Check: robot-only plan  Move(A,B), Move(B,C)",
        [move("A", "B"), move("B", "C")])

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for row in summary:
        print(" | ".join(row))
