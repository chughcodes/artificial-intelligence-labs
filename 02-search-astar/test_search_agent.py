"""
Systematic tests for the A* / BFS warehouse agent (Task 3).

Every test uses a case whose correct answer is known in advance, or is
checked against an independent method.

Run:
    python -m unittest test_search_agent -v
"""

import os
import random
import unittest
from collections import deque

from search_agent import (LAB_MAP, parse_map, astar, bfs, manhattan, zero,
                          euclidean, manhattan_x2, HEURISTICS)

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "maps", name)) as f:
        return parse_map(f.read())


def true_distance(grid):
    """Independent shortest-path length (simple BFS written separately)."""
    dist = {grid.start: 0}
    q = deque([grid.start])
    while q:
        r, c = q.popleft()
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if grid.is_free(n) and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist.get(grid.goal)


class PathValidityMixin:
    def assertValidPath(self, grid, result):
        self.assertTrue(result.found)
        self.assertEqual(result.path[0], grid.start)
        self.assertEqual(result.path[-1], grid.goal)
        for s in result.path:
            self.assertTrue(grid.is_free(s), f"path enters obstacle {s}")
        for a, b in zip(result.path, result.path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1,
                             f"illegal move {a}->{b}")
        self.assertEqual(len(result.actions), result.path_length)


class Test1OriginalWarehouse(unittest.TestCase, PathValidityMixin):
    def test_astar_finds_shortest_path(self):
        g = parse_map(LAB_MAP)
        r = astar(g, manhattan)
        self.assertValidPath(g, r)
        self.assertEqual(r.path_length, 40)
        self.assertEqual(r.path_length, true_distance(g))
        self.assertEqual(r.expanded, 63)

    def test_map_file_matches_lab_map(self):
        self.assertEqual(load("warehouse.txt").rows, parse_map(LAB_MAP).rows)


class Test2Trivial(unittest.TestCase, PathValidityMixin):
    def test_one_step_solution(self):
        g = load("test2_trivial.txt")
        for r in (astar(g, manhattan), bfs(g)):
            self.assertValidPath(g, r)
            self.assertEqual(r.path_length, 1)
            self.assertEqual(r.actions, ["Right"])
            self.assertEqual(r.expanded, 1)   # only S is expanded


class Test3NoSolution(unittest.TestCase):
    def test_reports_failure_and_terminates(self):
        g = load("test3_no_solution.txt")
        # Count the cells reachable from S independently (5 + 1 + 3 = 9)
        reachable = {g.start}
        q = deque([g.start])
        while q:
            r, c = q.popleft()
            for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if g.is_free(n) and n not in reachable:
                    reachable.add(n)
                    q.append(n)
        self.assertEqual(len(reachable), 9)
        self.assertNotIn(g.goal, reachable)
        for r in [bfs(g)] + [astar(g, h) for h in HEURISTICS.values()]:
            self.assertFalse(r.found)
            self.assertIsNone(r.path_length)
            # Each reachable cell is expanded exactly once, then the search
            # stops with an empty frontier: no infinite loop.
            self.assertEqual(r.expanded, len(reachable))

    def test_goal_completely_walled_in(self):
        g = parse_map("#######\n#S...##\n#####G#\n#######")
        self.assertFalse(astar(g, manhattan).found)


class Test4AlternativePaths(unittest.TestCase, PathValidityMixin):
    def test_two_routes_shortest_chosen(self):
        # Route over the top = 14 moves; route along the corridor = 16 moves
        g = load("test4_alternative_paths.txt")
        r = astar(g, manhattan)
        self.assertValidPath(g, r)
        self.assertEqual(r.path_length, 14)
        self.assertEqual(r.path_length, true_distance(g))
        self.assertEqual(r.path[1], (2, 1))   # it goes UP first, away from G

    def test_many_equal_shortest_paths(self):
        g = load("test4b_many_equal_paths.txt")
        r = astar(g, manhattan)
        self.assertValidPath(g, r)
        self.assertEqual(r.path_length, true_distance(g))


class TestRandomisedAgainstIndependentBFS(unittest.TestCase, PathValidityMixin):
    """500 random warehouses: every admissible A* must match the true distance."""

    def test_random_maps(self):
        rng = random.Random(42)
        solvable = 0
        for _ in range(500):
            h, w = rng.randint(4, 10), rng.randint(4, 14)
            cells = [["#"] * w] + [
                ["#"] + ["#" if rng.random() < 0.3 else "." for _ in range(w - 2)] + ["#"]
                for _ in range(h - 2)] + [["#"] * w]
            free = [(r, c) for r in range(h) for c in range(w) if cells[r][c] == "."]
            if len(free) < 2:
                continue
            s, goal = rng.sample(free, 2)
            cells[s[0]][s[1]], cells[goal[0]][goal[1]] = "S", "G"
            g = parse_map("\n".join("".join(r) for r in cells))
            d = true_distance(g)
            for res in (bfs(g), astar(g, manhattan), astar(g, zero), astar(g, euclidean)):
                if d is None:
                    self.assertFalse(res.found)
                else:
                    self.assertValidPath(g, res)
                    self.assertEqual(res.path_length, d)
            w2 = astar(g, manhattan_x2)
            if d is not None:
                solvable += 1
                self.assertValidPath(g, w2)
                # Weighted A* (w = 2) is bounded: cost <= 2 x optimal
                self.assertLessEqual(w2.path_length, 2 * d)
        self.assertGreater(solvable, 100)


class TestHeuristicProperties(unittest.TestCase):
    def test_manhattan_admissible_on_lab_map(self):
        g = parse_map(LAB_MAP)
        # exact distance to the goal from every cell (BFS from the goal)
        dist = {g.goal: 0}
        q = deque([g.goal])
        while q:
            s = q.popleft()
            for _, n, _ in g.successors(s):
                if n not in dist:
                    dist[n] = dist[s] + 1
                    q.append(n)
        for s, hstar in dist.items():
            self.assertLessEqual(manhattan(s, g.goal), hstar)
            self.assertLessEqual(euclidean(s, g.goal), hstar)

    def test_manhattan_x2_not_admissible(self):
        g = parse_map(LAB_MAP)
        # next to the goal: true cost 1, 2 x manhattan = 2  > 1
        self.assertGreater(manhattan_x2((6, 15), g.goal), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
