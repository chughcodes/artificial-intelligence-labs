"""
Tests for the warehouse goal-based agent.

Run:
    python -m unittest test_warehouse_agent -v
"""

import unittest
from collections import deque

from warehouse_agent import Warehouse, GoalBasedAgent, WAREHOUSE_MAP, ACTIONS


def independent_shortest_length(env):
    """A separate, plain BFS used only to cross-check the agent's answer."""
    dist = {env.start: 0}
    q = deque([env.start])
    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS.values():
            n = (r + dr, c + dc)
            if env.is_free(n) and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist.get(env.goal)


class TestWarehouseAgent(unittest.TestCase):

    def setUp(self):
        self.env = Warehouse(WAREHOUSE_MAP)
        self.agent = GoalBasedAgent(self.env)

    def test_start_and_goal_located(self):
        self.assertEqual(self.env.start, (1, 1))
        self.assertEqual(self.env.goal, (1, 19))

    def test_path_starts_at_S_and_ends_at_G(self):
        _, positions = self.agent.run(verbose=False)
        self.assertEqual(positions[0], self.env.start)
        self.assertEqual(positions[-1], self.env.goal)

    def test_path_is_collision_free(self):
        _, positions = self.agent.run(verbose=False)
        for pos in positions:
            self.assertTrue(self.env.is_free(pos), f"{pos} is an obstacle")

    def test_every_move_is_one_square(self):
        _, positions = self.agent.run(verbose=False)
        for a, b in zip(positions, positions[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_path_is_shortest(self):
        actions, _ = self.agent.run(verbose=False)
        self.assertEqual(len(actions), independent_shortest_length(self.env))
        self.assertEqual(len(actions), 20)

    def test_agent_state_reaches_goal(self):
        self.agent.run(verbose=False)
        self.assertEqual(self.agent.state, self.env.goal)

    def test_no_path_reports_failure(self):
        blocked = """\
#######
#S.#.G#
#..#..#
#######"""
        agent = GoalBasedAgent(Warehouse(blocked))
        self.assertIsNone(agent.run(verbose=False))

    def test_start_equals_goal_edge_case(self):
        agent = GoalBasedAgent(Warehouse("####\n#SG#\n####"))
        agent.goal = agent.state            # pretend the goal is already met
        actions, positions = agent.search()
        self.assertEqual(actions, [])

    def test_doubled_warehouse_still_solved(self):
        """'Think About It': a warehouse twice as large in each dimension."""
        rows = WAREHOUSE_MAP.splitlines()
        inner = [r[1:-1] for r in rows[1:-1]]
        # Tile the interior 2x2, keep only one S and one G, re-add walls.
        tiled = []
        for block in range(2):
            for line in inner:
                if block == 0:
                    left, right = line.replace("G", "."), line.replace("S", ".").replace("G", ".")
                else:
                    left, right = line.replace("S", ".").replace("G", "."), line.replace("S", ".")
                tiled.append("#" + left + right + "#")
        width = len(tiled[0])
        big = "\n".join(["#" * width] + tiled + ["#" * width])
        env = Warehouse(big)
        agent = GoalBasedAgent(env)
        actions, positions = agent.run(verbose=False)
        self.assertIsNotNone(actions)
        self.assertEqual(len(actions), independent_shortest_length(env))


if __name__ == "__main__":
    unittest.main(verbosity=2)
