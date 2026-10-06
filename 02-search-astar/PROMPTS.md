# Appendix: LLM prompts used (Claude)

These are the important prompts used during the laboratory, in order.

## Prompt 1: generate A* (Task 2)

> I am implementing a simple goal-based search agent in Python.
> The environment is a grid represented by an ASCII map. The agent starts at S and must reach G. The symbols # represent obstacles and . represents free cells. The agent can move up, down, left, or right, and every movement has cost 1.
> Implement A* search. Use Manhattan distance as the heuristic: h(n) = |x − x_G| + |y − y_G|.
>
> The program should:
> - represent grid positions as states, using a (row, col) tuple;
> - represent the warehouse as a list of strings, with a function that tells whether a cell is free;
> - maintain an appropriate frontier (a priority queue ordered by f);
> - calculate g(n), h(n) and f(n) explicitly;
> - avoid repeatedly expanding the same state;
> - reconstruct the path when the goal is reached, using parent pointers;
> - report whether a solution was found, the path and its length, and the number of states expanded;
> - report failure clearly if no path exists.
>
> Here is the map:
>
> ```
> #################
> #S....#.........#
> #.###.#.#######.#
> #...#.#.......#.#
> ###.#.#######.#.#
> #...#.........#.#
> #.###########.#.#
> #.............#G#
> #################
> ```
>
> Keep the implementation simple, use only the Python standard library, and explain the main components of the code.

## Prompt 2: BFS version (Task 5)

> Add a breadth-first search version of the same agent that uses the same Grid class, the same successor function and the same result object. Count "states expanded" in exactly the same way as in A* (a state is counted when it is removed from the frontier and its successors are generated; the goal test happens on removal), so that the two numbers can be compared fairly. Do not change the warehouse.

## Prompt 3: pluggable heuristics (Task 6)

> Make the heuristic a parameter of A*. Provide four heuristics: Manhattan distance, h(n) = 0, Euclidean distance, and 2 × Manhattan distance. Add command-line options to choose the algorithm, heuristic and map file.

## Prompt 4: explanation of the heuristic (Task 6)

> Explain why Manhattan distance is an appropriate heuristic for this warehouse when the robot can move only horizontally and vertically.

**Summary of the LLM's answer:** with only horizontal and vertical moves of cost 1, any path from n to G must make at least |dx| horizontal moves and at least |dy| vertical moves. So the true cost h*(n) ≥ |dx| + |dy|, which makes Manhattan distance admissible. Obstacles can only make the true path longer, never shorter. Manhattan distance is also consistent: one move changes it by exactly 1, which equals the step cost, so f never decreases along a path. It is the exact cost on an empty grid, which makes it the tightest simple admissible estimate for 4-connected movement. Euclidean distance is also admissible but smaller (looser), and would be the natural choice only if diagonal or free-direction movement were allowed.

(This explanation was then *checked experimentally*, not just accepted. See `test_manhattan_admissible_on_lab_map`, which verifies h(n) ≤ h*(n) for every reachable cell of the warehouse, and the Task 6 tables.)

## Prompt 5: tests (Task 3)

> Write unittest tests for the four required cases: (1) the original warehouse, (2) a trivial map #####/#SG##/##### where the goal is adjacent, (3) a map where the goal is inaccessible, checking failure is reported and the search terminates, and (4) a map with two different routes, checking the shortest one is returned. Check path length against an independent BFS written separately inside the test file, not against the code under test. Also add a randomised test over many random maps.

## Prompt 6: fixing a failing test

> The test test_reports_failure_and_terminates fails with "AssertionError: 9 != 7". Here is the map: #######/#S....#/###.###/#...#G#/#######. Is the bug in the search or in the test?

**LLM's answer:** the bug was in the test. The map has 5 + 1 + 3 = 9 reachable cells, not 7. The test was changed to count the reachable cells independently instead of using a hard-coded constant.
