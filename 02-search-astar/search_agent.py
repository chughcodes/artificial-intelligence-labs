"""
Search and A*: warehouse robot navigation
=========================================

Search problem  P = (S, A, T, s0, G, c)
    S  : all free grid cells (row, col)
    A  : {Up, Down, Left, Right}
    T  : T((r, c), a) = (r + dr, c + dc) if that cell is free, else undefined
    s0 : the position of 'S'
    G  : {position of 'G'}
    c  : 1 for every move

This module provides
    * parse_map()  - turns an ASCII map into a Grid
    * astar()      - A* search with a pluggable heuristic
    * bfs()        - breadth-first (blind) search for comparison
    * heuristics   - manhattan, zero, euclidean, and weighted (2 x manhattan)

Each search returns a SearchResult with:
    found, path (list of states), path_length (number of moves),
    expanded (number of states expanded), generated (states added to frontier)

Counting convention (identical for BFS and A*, so the numbers are comparable):
    a state counts as "expanded" when it is removed from the frontier and
    its successors are generated.  The goal test is performed when a state
    is removed from the frontier; the goal state itself is not counted as
    expanded.

Run:
    python search_agent.py               # A* on the laboratory map
    python search_agent.py --algo bfs    # BFS on the laboratory map
    python search_agent.py --map file.txt --algo astar --heuristic zero
"""

import argparse
import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# The laboratory map
# ---------------------------------------------------------------------------
LAB_MAP = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

# Action name -> (d_row, d_col).  The order fixes the order successors are
# generated, which only matters for tie-breaking.
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}
STEP_COST = 1


# ---------------------------------------------------------------------------
# Problem representation
# ---------------------------------------------------------------------------
class Grid:
    """The warehouse: a list of strings plus the start and goal positions."""

    def __init__(self, rows):
        self.rows = rows
        self.height = len(rows)
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, ch):
        for r, line in enumerate(self.rows):
            c = line.find(ch)
            if c != -1:
                return (r, c)
        raise ValueError(f"map has no '{ch}'")

    def is_free(self, state):
        """A state is valid if it is inside the map and not an obstacle."""
        r, c = state
        return 0 <= r < self.height and 0 <= c < len(self.rows[r]) \
            and self.rows[r][c] != "#"

    # ---- ACTIONS + TRANSITION ------------------------------------------
    def successors(self, state):
        """Yield (action, next_state, step_cost) for every valid action."""
        r, c = state
        for action, (dr, dc) in ACTIONS.items():
            nxt = (r + dr, c + dc)              # transition function T
            if self.is_free(nxt):               # invalid if wall/shelf/outside
                yield action, nxt, STEP_COST

    # ---- GOAL TEST --------------------------------------------------------
    def is_goal(self, state):
        return state == self.goal

    def render(self, path):
        g = [list(line) for line in self.rows]
        for r, c in path:
            if g[r][c] == ".":
                g[r][c] = "*"
        return "\n".join("".join(line) for line in g)


def parse_map(text):
    return Grid([line for line in text.strip("\n").splitlines()])


# ---------------------------------------------------------------------------
# Heuristics  h(n): estimated cost from n to the goal
# ---------------------------------------------------------------------------
def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def zero(state, goal):
    return 0


def euclidean(state, goal):
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def manhattan_x2(state, goal):
    return 2 * manhattan(state, goal)


HEURISTICS = {
    "manhattan": manhattan,
    "zero": zero,
    "euclidean": euclidean,
    "manhattan_x2": manhattan_x2,
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    algorithm: str
    found: bool
    path: list = field(default_factory=list)       # list of states s0..goal
    actions: list = field(default_factory=list)    # list of action names
    expanded: int = 0
    generated: int = 0

    @property
    def path_length(self):
        """Number of moves (= path cost, because every move costs 1)."""
        return len(self.path) - 1 if self.found else None

    def report(self, grid=None):
        lines = [f"Algorithm       : {self.algorithm}",
                 f"Solution found  : {'yes' if self.found else 'no'}"]
        if self.found:
            lines += [f"Path length     : {self.path_length}",
                      f"States expanded : {self.expanded}",
                      f"Path (states)   : {self.path}",
                      f"Actions         : {' '.join(self.actions) or '(none)'}"]
            if grid is not None:
                lines += ["", grid.render(self.path)]
        else:
            lines += [f"States expanded : {self.expanded}",
                      "No path exists from S to G."]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PATH RECONSTRUCTION
# ---------------------------------------------------------------------------
def reconstruct(came_from, state):
    """Follow parent pointers back from the goal to the start."""
    path, actions = [state], []
    while came_from[state] is not None:
        parent, action = came_from[state]
        actions.append(action)
        path.append(parent)
        state = parent
    return path[::-1], actions[::-1]


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------
def astar(grid, heuristic=manhattan, name=None):
    """
    A* graph search.

    Frontier: a binary heap (priority queue) of entries
        (f, h, tie, g, state)
    ordered by f(n) = g(n) + h(n); ties on f are broken by smaller h (prefer
    nodes that look closer to the goal) and then by insertion order.

    best_g[state] holds the cheapest g found so far.  A heap entry whose g
    is worse than best_g is stale and is skipped when popped.  This, plus
    the closed set, prevents a state from being expanded repeatedly.
    """
    start, goal = grid.start, grid.goal
    tie = itertools.count()

    g0 = 0
    h0 = heuristic(start, goal)
    frontier = [(g0 + h0, h0, next(tie), g0, start)]     # FRONTIER
    best_g = {start: 0}                                    # cheapest g(n)
    came_from = {start: None}                              # parent pointers
    closed = set()                                         # VISITED / expanded
    expanded = 0
    generated = 1

    while frontier:
        f, h, _, g, state = heapq.heappop(frontier)        # lowest f(n)
        if state in closed or g > best_g[state]:
            continue                                       # stale entry
        if grid.is_goal(state):                            # GOAL TEST
            path, actions = reconstruct(came_from, state)
            return SearchResult(name or f"A* ({heuristic.__name__})",
                                True, path, actions, expanded, generated)
        closed.add(state)
        expanded += 1
        for action, nxt, cost in grid.successors(state):
            g_new = g + cost                               # g(n)
            if nxt in closed and g_new >= best_g[nxt]:
                continue
            if g_new < best_g.get(nxt, math.inf):
                best_g[nxt] = g_new
                came_from[nxt] = (state, action)
                closed.discard(nxt)        # allow re-opening if improved
                h_new = heuristic(nxt, goal)               # h(n)
                f_new = g_new + h_new                      # f(n) = g + h
                heapq.heappush(frontier, (f_new, h_new, next(tie), g_new, nxt))
                generated += 1

    return SearchResult(name or f"A* ({heuristic.__name__})",
                        False, expanded=expanded, generated=generated)


# ---------------------------------------------------------------------------
# Breadth-first search (blind search) for comparison
# ---------------------------------------------------------------------------
def bfs(grid):
    """
    BFS graph search.  Frontier is a FIFO queue; `came_from` doubles as the
    reached set, so each state enters the frontier at most once.
    """
    start = grid.start
    frontier = deque([start])
    came_from = {start: None}
    expanded = 0
    generated = 1

    while frontier:
        state = frontier.popleft()
        if grid.is_goal(state):
            path, actions = reconstruct(came_from, state)
            return SearchResult("BFS", True, path, actions, expanded, generated)
        expanded += 1
        for action, nxt, _ in grid.successors(state):
            if nxt not in came_from:
                came_from[nxt] = (state, action)
                frontier.append(nxt)
                generated += 1

    return SearchResult("BFS", False, expanded=expanded, generated=generated)


def solve(grid, algo="astar", heuristic="manhattan"):
    if algo == "bfs":
        return bfs(grid)
    return astar(grid, HEURISTICS[heuristic])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A* / BFS warehouse search")
    ap.add_argument("--map", help="path to an ASCII map file (default: lab map)")
    ap.add_argument("--algo", choices=["astar", "bfs"], default="astar")
    ap.add_argument("--heuristic", choices=list(HEURISTICS), default="manhattan")
    args = ap.parse_args()

    text = open(args.map).read() if args.map else LAB_MAP
    grid = parse_map(text)
    print(solve(grid, args.algo, args.heuristic).report(grid))
