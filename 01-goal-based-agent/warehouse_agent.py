"""
Goal-Based Agent for the Warehouse Navigation Problem
=====================================================

An autonomous warehouse vehicle must travel from the loading bay (S) to the
dispatch area (G) without crossing shelving units (#).

Agent architecture (goal-based agent, as in the lecture)
-------------------------------------------------------
    Environment  -> the 2-D warehouse grid (static, fully observable)
    Sensors      -> perceive(): read the vehicle position and the map
    State        -> current (row, col) position of the vehicle
    Goal         -> the (row, col) position of G
    Actions      -> Up, Down, Left, Right (one grid square per move)
    Decision     -> a search component that asks "what sequence of actions
                    will take me from my current state to my goal?" and then
                    executes the first action of that plan, repeatedly.

Search algorithm chosen: Breadth-First Search (BFS)
---------------------------------------------------
* Every move costs exactly 1, so the shortest path is the path with the
  fewest moves.  BFS expands states in order of their distance from S, so
  the first time it reaches G it has found a shortest path (BFS is optimal
  for uniform step costs).
* BFS is complete: on a finite grid it will always find a path if one
  exists, and it terminates with "no path" if none exists.
* The grid is small (7 x 21 = 147 cells, ~70 of them free) so BFS's
  memory cost of O(number of states) is no concern.
* A visited set ensures each cell is expanded at most once, so the agent
  can never loop.

Run:
    python warehouse_agent.py
"""

from collections import deque

# ---------------------------------------------------------------------------
# The warehouse map given in the lab sheet
# ---------------------------------------------------------------------------
WAREHOUSE_MAP = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# Actions: name -> (change in row, change in column)
ACTIONS = {
    "Up":    (-1, 0),
    "Down":  (1, 0),
    "Left":  (0, -1),
    "Right": (0, 1),
}

OBSTACLE, FREE, START, GOAL = "#", ".", "S", "G"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Warehouse:
    """The environment: a 2-D grid of characters."""

    def __init__(self, map_text):
        self.grid = [list(line) for line in map_text.strip("\n").splitlines()]
        self.rows = len(self.grid)
        self.cols = max(len(r) for r in self.grid)
        self.start = self._find(START)
        self.goal = self._find(GOAL)

    def _find(self, symbol):
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == symbol:
                    return (r, c)
        raise ValueError(f"Symbol '{symbol}' not found in the map")

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < len(self.grid[r])

    def is_free(self, pos):
        """A cell can be entered if it is inside the map and not a shelf."""
        r, c = pos
        return self.in_bounds(pos) and self.grid[r][c] != OBSTACLE

    def render(self, path=None):
        """Return the map as text, marking the path with '*'."""
        g = [row[:] for row in self.grid]
        for (r, c) in (path or []):
            if g[r][c] == FREE:
                g[r][c] = "*"
        return "\n".join("".join(row) for row in g)


# ---------------------------------------------------------------------------
# Goal-based agent
# ---------------------------------------------------------------------------
class GoalBasedAgent:
    """
    Keeps an internal state (its position), knows its goal, and uses a
    model of how actions change the state (the transition function) to
    search for an action sequence that achieves the goal.
    """

    def __init__(self, environment):
        self.env = environment
        self.state = environment.start      # internal state: current position
        self.goal = environment.goal        # explicit objective
        self.nodes_expanded = 0

    # ----- model of the world -------------------------------------------
    def result(self, state, action):
        """Transition model: the state reached by doing `action` in `state`."""
        dr, dc = ACTIONS[action]
        return (state[0] + dr, state[1] + dc)

    def applicable_actions(self, state):
        """Actions that do not drive the vehicle into a shelf or a wall."""
        return [a for a in ACTIONS if self.env.is_free(self.result(state, a))]

    def goal_test(self, state):
        return state == self.goal

    # ----- decision-making component: Breadth-First Search -------------
    def search(self):
        """
        Return (list_of_actions, list_of_positions) for a shortest path from
        the current state to the goal, or (None, None) if none exists.
        """
        start = self.state
        if self.goal_test(start):
            return [], [start]

        frontier = deque([start])           # FIFO queue
        parent = {start: (None, None)}      # also serves as the visited set
        self.nodes_expanded = 0

        while frontier:
            state = frontier.popleft()
            self.nodes_expanded += 1
            for action in self.applicable_actions(state):
                child = self.result(state, action)
                if child in parent:          # already discovered
                    continue
                parent[child] = (state, action)
                if self.goal_test(child):    # goal test on generation
                    return self._reconstruct(parent, child)
                frontier.append(child)
        return None, None                    # frontier empty -> no path

    @staticmethod
    def _reconstruct(parent, node):
        actions, positions = [], [node]
        while parent[node][0] is not None:
            prev, action = parent[node]
            actions.append(action)
            positions.append(prev)
            node = prev
        return actions[::-1], positions[::-1]

    # ----- agent loop: plan, then act --------------------------------
    def run(self, verbose=True):
        actions, positions = self.search()
        if actions is None:
            if verbose:
                print("No collision-free path exists from S to G.")
            return None
        # Execute the plan, updating the internal state after every action
        for action in actions:
            nxt = self.result(self.state, action)
            assert self.env.is_free(nxt), "Plan would cause a collision!"
            self.state = nxt
        assert self.goal_test(self.state)
        if verbose:
            print(f"Path found: {len(actions)} moves, "
                  f"{self.nodes_expanded} states expanded\n")
            print("Actions:", " -> ".join(actions), "\n")
            print("Positions (row, col):", positions, "\n")
            print(self.env.render(positions))
        return actions, positions


if __name__ == "__main__":
    print("=== Warehouse Navigation: Goal-Based Agent (BFS) ===\n")
    warehouse = Warehouse(WAREHOUSE_MAP)
    print(f"Grid size: {warehouse.rows} x {warehouse.cols}")
    print(f"Start S = {warehouse.start}, Goal G = {warehouse.goal}\n")
    GoalBasedAgent(warehouse).run()
