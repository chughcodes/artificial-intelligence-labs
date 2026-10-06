"""
Unit tests for the planner (Task 3), plus checks of the logic in isolation.

Run:
    python -m unittest test_planner -v
"""

import unittest
from itertools import product

from planner import (INITIAL, GOAL, warehouse_actions, make_action, move,
                     pickup, drop, applicable, apply, goal_satisfied,
                     bfs_plan, validate_plan)
from run_tests import irrelevant_actions


def by_name(actions, name):
    return next(a for a in actions if a.name == name)


class TestLogic(unittest.TestCase):
    """Task 0: applicability of actions in I, using the preconditions."""

    def test_pickup_applicable_initially(self):
        self.assertTrue(applicable(INITIAL, pickup("A")))

    def test_drop_at_c_not_applicable_initially(self):
        self.assertFalse(applicable(INITIAL, drop("C")))

    def test_only_two_actions_applicable_initially(self):
        names = {a.name for a in warehouse_actions() if applicable(INITIAL, a)}
        self.assertEqual(names, {"Move(A,B)", "PickUp(Package,A)"})

    def test_no_direct_move_a_to_c(self):
        self.assertNotIn("Move(A,C)", {a.name for a in warehouse_actions()})

    def test_apply_removes_then_adds(self):
        s = apply(INITIAL, pickup("A"))
        self.assertEqual(s, frozenset({"At(Robot,A)", "Holding(Package)"}))

    def test_apply_refuses_inapplicable_action(self):
        with self.assertRaises(ValueError):
            apply(INITIAL, drop("C"))

    def test_negative_precondition(self):
        a = make_action("X", neg_pre=["Holding(Package)"], pos_eff=["Done"])
        self.assertTrue(applicable(INITIAL, a))
        self.assertFalse(applicable(INITIAL | {"Holding(Package)"}, a))

    def test_delete_before_add(self):
        # If a proposition is in both lists it must end up TRUE
        a = make_action("Refresh", pos_eff=["P"], neg_eff=["P"])
        self.assertIn("P", apply(frozenset({"P"}), a))

    def test_goal_not_satisfied_by_robot_at_c(self):
        self.assertFalse(goal_satisfied(
            frozenset({"At(Robot,C)", "At(Package,A)"}), GOAL))


class TestA_Solvable(unittest.TestCase):
    def test_plan_found_valid_and_shortest(self):
        plan, states, _ = bfs_plan(INITIAL, GOAL, warehouse_actions())
        self.assertEqual([a.name for a in plan],
                         ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)",
                          "Drop(Package,C)"])
        self.assertTrue(validate_plan(INITIAL, GOAL, plan)[0])
        self.assertTrue(goal_satisfied(states[-1], GOAL))

    def test_no_shorter_plan_exists(self):
        """Brute force: no sequence of 1-3 actions achieves the goal."""
        acts = warehouse_actions()
        for n in range(1, 4):
            for seq in product(acts, repeat=n):
                self.assertFalse(validate_plan(INITIAL, GOAL, list(seq))[0])

    def test_every_state_transition_is_consistent(self):
        plan, states, _ = bfs_plan(INITIAL, GOAL, warehouse_actions())
        for a, s, s2 in zip(plan, states, states[1:]):
            self.assertTrue(applicable(s, a))
            self.assertEqual(apply(s, a), s2)


class TestB_Impossible(unittest.TestCase):
    def test_no_pickup_no_plan(self):
        plan, _, _ = bfs_plan(INITIAL, GOAL, warehouse_actions(include_pickup=False))
        self.assertIsNone(plan)

    def test_unreachable_location_no_plan(self):
        acts = [move("A", "B"), move("B", "A")] + \
               [pickup(l) for l in "ABC"] + [drop(l) for l in "ABC"]
        self.assertIsNone(bfs_plan(INITIAL, GOAL, acts)[0])


class TestC_IrrelevantActions(unittest.TestCase):
    def test_irrelevant_actions_do_not_fool_planner(self):
        acts = warehouse_actions() + irrelevant_actions()
        plan, states, _ = bfs_plan(INITIAL, GOAL, acts)
        self.assertTrue(validate_plan(INITIAL, GOAL, plan)[0])
        self.assertEqual(len(plan), 4)
        # The Shuttle(A,C) shortcut gets the ROBOT to C in one step, but that
        # state must not be accepted as a goal state.
        robot_at_c = apply(INITIAL, by_name(acts, "Shuttle(A,C)"))
        self.assertIn("At(Robot,C)", robot_at_c)
        self.assertFalse(goal_satisfied(robot_at_c, GOAL))
        self.assertNotIn("Shuttle(A,C)", [a.name for a in plan])

    def test_robot_only_plan_rejected(self):
        ok, _ = validate_plan(INITIAL, GOAL, [move("A", "B"), move("B", "C")])
        self.assertFalse(ok)

    def test_robot_starts_at_c(self):
        init = frozenset({"At(Robot,C)", "At(Package,A)"})
        plan, _, _ = bfs_plan(init, GOAL, warehouse_actions())
        self.assertEqual(len(plan), 6)          # must go back to A first
        self.assertTrue(validate_plan(init, GOAL, plan)[0])


class TestHandPlans(unittest.TestCase):
    def test_sheet_example_sequence_is_invalid(self):
        plan = [move("A", "B"), pickup("B"), move("B", "C"), drop("C")]
        ok, lines = validate_plan(INITIAL, GOAL, plan)
        self.assertFalse(ok)
        self.assertIn("PickUp(Package,B) NOT applicable", lines[-1])

    def test_hand_constructed_plan_is_valid(self):
        plan = [pickup("A"), move("A", "B"), move("B", "C"), drop("C")]
        self.assertTrue(validate_plan(INITIAL, GOAL, plan)[0])

    def test_goal_already_true_gives_empty_plan(self):
        init = frozenset({"At(Robot,A)", "At(Package,C)"})
        plan, states, _ = bfs_plan(init, GOAL, warehouse_actions())
        self.assertEqual(plan, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
