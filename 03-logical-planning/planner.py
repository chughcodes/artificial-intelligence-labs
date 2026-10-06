"""
A simple STRIPS-style planning agent: Logic + Search = Planning
==============================================================

* A STATE is a frozenset of ground propositions (strings), e.g.
      frozenset({"At(Robot,A)", "At(Package,A)"})
  Closed-world assumption: any proposition not in the set is false.

* An ACTION has
      name, positive preconditions, negative preconditions,
      positive effects (add list), negative effects (delete list).

* LOGIC  - applicable(S, a):  S |= Preconditions(a)
           i.e. every positive precondition is in S and no negative
           precondition is in S.
* EFFECTS - apply(S, a) = (S - neg_effects(a)) | pos_effects(a)
            (negative effects removed first, then positive effects added)
* GOAL   - goal_satisfied(S, G): every goal proposition is in S (G subset of S)
* SEARCH - breadth-first search over states; returns a shortest plan
           (fewest actions) or None if no plan exists.

Run:
    python planner.py
"""

from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: frozenset = field(default_factory=frozenset)
    neg_pre: frozenset = field(default_factory=frozenset)
    pos_eff: frozenset = field(default_factory=frozenset)
    neg_eff: frozenset = field(default_factory=frozenset)

    def __str__(self):
        return self.name


def make_action(name, pos_pre=(), neg_pre=(), pos_eff=(), neg_eff=()):
    return Action(name, frozenset(pos_pre), frozenset(neg_pre),
                  frozenset(pos_eff), frozenset(neg_eff))


# ---------------------------------------------------------------------------
# LOGIC: is the action applicable?   S |= Preconditions(a)
# ---------------------------------------------------------------------------
def applicable(state, action):
    return action.pos_pre <= state and not (action.neg_pre & state)


def unsatisfied_preconditions(state, action):
    """Explain *why* an action is not applicable (used for reporting)."""
    missing = [f"{p} is false" for p in sorted(action.pos_pre - state)]
    violated = [f"{p} is true (must be false)" for p in sorted(action.neg_pre & state)]
    return missing + violated


# ---------------------------------------------------------------------------
# EFFECTS: S' = Apply(S, a)
# ---------------------------------------------------------------------------
def apply(state, action):
    if not applicable(state, action):
        raise ValueError(f"{action.name} is not applicable")
    return frozenset((state - action.neg_eff) | action.pos_eff)   # 1. delete  2. add


# ---------------------------------------------------------------------------
# GOAL
# ---------------------------------------------------------------------------
def goal_satisfied(state, goal):
    return goal <= state


# ---------------------------------------------------------------------------
# SEARCH: breadth-first search over states
# ---------------------------------------------------------------------------
def bfs_plan(initial, goal, actions):
    """
    Returns (plan, states, expanded) where plan is a list of Actions and
    states is [S0, S1, ..., Sn]; or (None, None, expanded) if no plan exists.
    """
    initial, goal = frozenset(initial), frozenset(goal)
    if goal_satisfied(initial, goal):
        return [], [initial], 0

    frontier = deque([initial])
    parent = {initial: None}               # also the visited set
    expanded = 0

    while frontier:
        state = frontier.popleft()
        expanded += 1
        for action in actions:
            if not applicable(state, action):          # logic
                continue
            nxt = apply(state, action)                 # effects
            if nxt in parent:                          # already reached
                continue
            parent[nxt] = (state, action)
            if goal_satisfied(nxt, goal):              # goal
                plan, states = [], [nxt]
                s = nxt
                while parent[s] is not None:
                    prev, a = parent[s]
                    plan.append(a)
                    states.append(prev)
                    s = prev
                return plan[::-1], states[::-1], expanded
            frontier.append(nxt)                       # search
    return None, None, expanded                        # no plan exists


# ---------------------------------------------------------------------------
# Independent plan checker (simulates a plan step by step)
# ---------------------------------------------------------------------------
def validate_plan(initial, goal, plan):
    """
    Re-executes the plan from scratch, checking every precondition and the
    goal.  Returns (is_valid, list_of_report_lines).
    """
    state = frozenset(initial)
    lines = []
    for i, action in enumerate(plan, 1):
        if not applicable(state, action):
            lines.append(f"Step {i}: {action.name} NOT applicable: "
                         + "; ".join(unsatisfied_preconditions(state, action)))
            return False, lines
        pre = ", ".join(sorted(action.pos_pre)) or "(none)"
        if action.neg_pre:
            pre += ", " + ", ".join("not " + p for p in sorted(action.neg_pre))
        state = apply(state, action)
        lines.append(f"Step {i}: {action.name}: preconditions [{pre}] hold -> "
                     f"{fmt_state(state)}")
    ok = goal_satisfied(state, frozenset(goal))
    lines.append("Goal " + ("SATISFIED" if ok else "NOT satisfied")
                 + f" in final state {fmt_state(state)}")
    return ok, lines


# ---------------------------------------------------------------------------
# The warehouse domain
# ---------------------------------------------------------------------------
LOCATIONS = ["A", "B", "C"]
CONNECTIONS = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]


def move(x, y):
    return make_action(f"Move({x},{y})",
                       pos_pre=[f"At(Robot,{x})"],
                       pos_eff=[f"At(Robot,{y})"],
                       neg_eff=[f"At(Robot,{x})"])


def pickup(loc, obj="Package"):
    return make_action(f"PickUp({obj},{loc})",
                       pos_pre=[f"At(Robot,{loc})", f"At({obj},{loc})"],
                       pos_eff=[f"Holding({obj})"],
                       neg_eff=[f"At({obj},{loc})"])


def drop(loc, obj="Package"):
    return make_action(f"Drop({obj},{loc})",
                       pos_pre=[f"At(Robot,{loc})", f"Holding({obj})"],
                       pos_eff=[f"At({obj},{loc})"],
                       neg_eff=[f"Holding({obj})"])


def warehouse_actions(include_pickup=True):
    acts = [move(x, y) for x, y in CONNECTIONS]
    if include_pickup:
        acts += [pickup(l) for l in LOCATIONS]
    acts += [drop(l) for l in LOCATIONS]
    return acts


INITIAL = frozenset({"At(Robot,A)", "At(Package,A)"})
GOAL = frozenset({"At(Package,C)"})


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def fmt_state(state):
    return "{" + ", ".join(sorted(state)) + "}"


def solve_and_report(title, initial, goal, actions):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("Initial state :", fmt_state(initial))
    print("Goal          :", fmt_state(goal))
    print("Actions       :", ", ".join(a.name for a in actions))
    plan, states, expanded = bfs_plan(initial, goal, actions)
    if plan is None:
        print(f"\nNo plan found  (states expanded: {expanded})\n")
        return None
    print(f"\nPlan found: {len(plan)} actions  (states expanded: {expanded})")
    print(f"  S0: {fmt_state(states[0])}")
    for i, (a, s) in enumerate(zip(plan, states[1:]), 1):
        print(f"  {a.name:<22} -> S{i}: {fmt_state(s)}")
    ok, lines = validate_plan(initial, goal, plan)
    print("\nIndependent check of every action:")
    for line in lines:
        print("  " + line)
    print("Plan valid:", ok, "\n")
    return plan


if __name__ == "__main__":
    # Which actions are applicable initially? (Task 0)
    print("Applicability in I =", fmt_state(INITIAL))
    for a in warehouse_actions():
        why = "" if applicable(INITIAL, a) else \
            "   because " + "; ".join(unsatisfied_preconditions(INITIAL, a))
        print(f"  {a.name:<20} {'APPLICABLE' if applicable(INITIAL, a) else 'not applicable'}{why}")
    print()

    solve_and_report("Original warehouse problem", INITIAL, GOAL, warehouse_actions())
