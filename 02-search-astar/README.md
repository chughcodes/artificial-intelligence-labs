# Lab 2: Search and A* (Using an LLM as an Engineering Assistant)

Warehouse robot navigation, solved with A* search and compared with breadth-first search.

## Files

| File | Purpose |
|---|---|
| `search_agent.py` | **Final program**: grid model, A* (pluggable heuristic), BFS, command-line interface |
| `test_search_agent.py` | Systematic tests (Tests 1–4, randomised check against an independent BFS, heuristic properties) |
| `experiments.py` | Runs every experiment and regenerates `results.md` |
| `results.md` | Full generated results: all maps, paths and tables |
| `maps/` | `warehouse.txt` (lab map), the test maps, and `open_warehouse.txt` |
| `PROMPTS.md` | Appendix: the prompts used with the LLM |
| `sample_output.txt`, `sample_output_bfs.txt`, `test_output.txt` | Saved program and test output |

## How to reproduce

Python 3 only, with no external libraries.

```bash
python search_agent.py                                   # A* (Manhattan) on the lab warehouse
python search_agent.py --algo bfs                        # BFS on the lab warehouse
python search_agent.py --heuristic zero                  # A* with h = 0 (also: euclidean, manhattan_x2)
python search_agent.py --map maps/test3_no_solution.txt  # any map file
python -m unittest test_search_agent -v                  # all tests
python experiments.py                                    # regenerate every table in results.md
```

**Counting convention**, used identically for BFS and A* so that the numbers are comparable: a state is *expanded* when it is removed from the frontier and its successors are generated. The goal test is performed on removal, and the goal itself is not counted. A* breaks ties on f by preferring smaller h, then insertion order.

---

## 1. Task 0: Formulation of the search problem

| Component | Specification |
|---|---|
| **State S** | The robot's position, a pair (row, col) of a free cell in the grid. The map is fixed, so the position is the whole state. The lab warehouse has 64 free cells, so \|S\| = 64. |
| **Actions A** | {Up, Down, Left, Right} |
| **Transition T** | T((r, c), Up) = (r−1, c); Down → (r+1, c); Left → (r, c−1); Right → (r, c+1). Defined only if the resulting cell is inside the map and is not `#`. Otherwise the action is not applicable. |
| **Initial state s0** | The position of `S`: (1, 1) |
| **Goal G** | {(7, 15)}, the position of `G`. Goal test: state == (7, 15). |
| **Cost c** | c(s, a, s′) = 1 for every move. The path cost is the number of moves. |

**(a) What information is necessary to specify a state?**
Only the robot's (row, col) coordinates. The map, the obstacles and the goal never change, so they belong to the *problem* and not to the state. (If the robot also had an orientation, a battery level or a carried package, those would have to be added to the state.)

**(b) What makes an action invalid?**
An action is invalid if the cell it leads to is an obstacle (`#`), or lies outside the map. Here the outer walls prevent the second case.

**(c) Is this a deterministic search problem?**
Yes. Each action has exactly one outcome (the robot always moves exactly one cell in the chosen direction). The map is fully known and static, and costs are fixed. So the result of any action sequence can be predicted exactly, and a plan can be computed fully in advance.

**(d) What would constitute a solution?**
A sequence of actions a1, …, ak that, starting at s0 and applying T at each step, ends in the goal state, with every intermediate state free. An *optimal* solution is a solution of minimum total cost (fewest moves).

---

## 2. Task 1: Design of the agent (written before prompting the LLM)

1. **State representation:** a Python tuple `(row, col)`. Tuples are immutable and hashable, so they can be used as dictionary keys and set members.
2. **Warehouse representation:** a list of strings, one per map row. `grid.rows[r][c]` gives the symbol. `S` and `G` are located once when the map is parsed.
3. **Valid actions:** a dictionary `ACTIONS = {"Up": (-1, 0), ...}`. A successor function adds each offset to the current state and keeps it only if the new cell is inside the map and not `#`. It yields `(action, next_state, cost=1)`.
4. **Goal recognition:** `state == goal`, tested when a state is **removed** from the frontier. For A* this is essential for optimality: testing on generation could accept a goal reached by a more expensive path before a cheaper one is found.
5. **Frontier contents:** for A*, a priority queue (binary heap) of `(f, h, tie_counter, g, state)`. f orders the queue, h breaks ties, the counter keeps ordering stable and avoids comparing states, and g is the cost so far. Alongside it, keep `best_g[state]` (cheapest known cost to each state) and a closed set of expanded states. For BFS, a FIFO queue of states.
6. **Path reconstruction:** a `came_from` dictionary that maps each state to `(parent_state, action)`. When the goal is reached, follow parents back to the start and reverse the list.

**Reported on termination:** whether a solution was found; the path (states and actions); the path length; the number of states expanded. The map is also printed with the path drawn as `*`.

---

## 3. The final Python program

See [`search_agent.py`](search_agent.py). Its main components are:
- `Grid`: the problem (map, start, goal, successor function, goal test);
- heuristics: `manhattan`, `zero`, `euclidean`, `manhattan_x2`;
- `astar(grid, heuristic)` and `bfs(grid)`, both returning a `SearchResult`;
- `reconstruct(came_from, state)`: path reconstruction.

Example output on the lab warehouse (`python search_agent.py`):

```
Algorithm       : A* (manhattan)
Solution found  : yes
Path length     : 40
States expanded : 63
Actions         : Right Right Right Right Down Down Down Down Right Right Right Right Right Right Right Right Up Up Left Left Left Left Left Left Up Up Right Right Right Right Right Right Right Right Down Down Down Down Down Down

#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

## 4. The prompts used with the LLM

See [`PROMPTS.md`](PROMPTS.md).

---

## 5. Task 3: Results of the tests

All tests are automated in `test_search_agent.py` (10 tests, all passing; see `test_output.txt`). The full program output for every test is in `results.md`.

| Test | Map | Path found | Path length | States expanded | Expected | Pass? |
|---|---|---|---|---|---|---|
| **1. Original warehouse** | `maps/warehouse.txt` | yes | **40** | **63** | shortest = 40 (independent BFS) | ✅ |
| **2. Trivial** | `#SG##` | yes | **1** (`Right`) | 1 | one-step solution | ✅ |
| **3. No solution** | sheet's example | **no** | – | 9 | failure reported; terminates after expanding the 9 reachable cells | ✅ |
| **4. Alternative paths** | `maps/test4_alternative_paths.txt` | yes | **14** | 24 | two routes, 14 (over the top) and 16 (along the corridor); must choose 14 | ✅ |
| 4b. Many equal shortest paths | `maps/test4b_many_equal_paths.txt` | yes | 10 | 16 | shortest = 10 | ✅ |
| Randomised | 500 random warehouses | – | – | – | BFS, A*(Manhattan), A*(0), A*(Euclidean) all match an independent BFS; A*(2×) ≤ 2 × optimal | ✅ |

**Test 1 path:** (1,1) → (1,5) → down to (5,5) → right to (5,13) → up to (3,13) → left to (3,7) → up to (1,7) → right to (1,15) → down to the goal (7,15). The actions are listed above.

**Test 4 map:** going straight right from S heads directly towards G but ends in a 16-move detour. Going *up* first, apparently away from G, gives the 14-move route. A* correctly goes up (`path[1] == (2, 1)` is asserted in the test).

```
#############
#***********#     <- 14-move route chosen by A*
#*#########*#
#S........#G#     <- corridor: leads to the 16-move route
#########.#.#
#########.#.#
#########...#
#############
```

---

## 6. Task 4: Inspecting the A* algorithm

| Concept | Where it appears in `search_agent.py` |
|---|---|
| **State** | A `(row, col)` tuple, e.g. `grid.start`; the `state` variable in `astar()` |
| **Action** | `ACTIONS` dictionary (`"Up": (-1, 0)`, …), iterated in `Grid.successors()` |
| **Transition** | `Grid.successors()`: `nxt = (r + dr, c + dc)`, kept only if `self.is_free(nxt)` |
| **Goal test** | `Grid.is_goal()`: `state == self.goal`, called in `astar()` right after a state is popped |
| **g(n)** | `g_new = g + cost` in `astar()`; stored in `best_g[nxt]` and in the heap entry |
| **h(n)** | `h_new = heuristic(nxt, goal)` (and `h0` for the start), computed by `manhattan()` etc. |
| **f(n)** | `f_new = g_new + h_new`, the first field of the heap entry |
| **Frontier** | `frontier`, a list managed with `heapq.heappush` / `heapq.heappop` |
| **Visited states** | `closed` set (expanded states) plus `best_g` dict (cheapest cost found so far) |
| **Path reconstruction** | `came_from` dict (parent + action), and the `reconstruct()` function |

**(a) What data structure is used for the A* frontier?**
A **binary min-heap** (priority queue) from Python's `heapq` module, holding tuples `(f, h, tie, g, state)`.

**(b) How does the program select the next state to expand?**
`heapq.heappop` returns the entry with the **smallest f(n)**. Ties are broken by smaller h(n), preferring states that appear closer to the goal, then by insertion order. If the popped entry is stale (the state is already closed, or a cheaper g has since been found for it), it is discarded and the next one is popped.

**(c) Where is the heuristic calculated?**
When a successor is generated (`h_new = heuristic(nxt, goal)`) and once for the start state (`h0`). The heuristic function itself is passed in as a parameter (`manhattan` by default), so Task 6 can swap it without changing A*.

**(d) Does the program explicitly calculate f(n) = g(n) + h(n)?**
Yes: `f_new = g_new + h_new` (and `g0 + h0` for the start). f is stored as the first element of each heap entry, which is what orders the priority queue.

**(e) How does the program prevent unnecessary repeated exploration?**
- A **closed set**: a state that has been expanded is not expanded again, unless a strictly cheaper path to it is found later. That can only happen with an inconsistent heuristic such as 2 × Manhattan.
- A **`best_g` dictionary**: a successor is pushed only if the new path to it is cheaper than any found so far.
- **Stale entry check**: when an outdated heap entry is popped (`g > best_g[state]`), it is skipped.

Together these mean each state is expanded at most once with a consistent heuristic. That is why the no-solution test terminates after exactly 9 expansions instead of looping.

---

## 7. Task 5: BFS / A* comparison (same warehouse, unchanged)

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | yes | yes |
| Path length | 40 | 40 |
| States expanded | **63** | **63** |
| States generated | 64 | 65 |

**(a) Did both algorithms find a solution?** Yes.

**(b) Did they find paths of the same length?** Yes, 40 moves each, which is the optimum. In fact they return the identical path. BFS is optimal because every step costs 1; A* is optimal because Manhattan distance is admissible (and consistent).

**(c) Which algorithm expanded fewer states?**
**Neither. On this warehouse both expanded 63 states.** That is every free cell except the goal itself (the map has 64 free cells, all reachable). This was surprising at first, so I investigated rather than assuming a bug:
- This warehouse is a **maze that defeats the Manhattan heuristic**. h(S) = 20, but the true cost is 40, because the only route winds up, down and back across the map.
- The worst part is the **bottom corridor** (row 7). It runs from the far left right up to (7,13), two squares from G, but is a dead end separated from G by a wall. Every cell in it has a small h, so its f = g + h is below the optimal cost of 40.
- A* **must** expand every state with f(n) < C\* = 40 when its heuristic is admissible. A separate calculation shows that **49** reachable cells have g(n) + h(n) < 40 and so must be expanded by *any* A* using Manhattan distance. The other **15** cells have f = 40 exactly, and *all 15 lie on the optimal path itself* (one of them is the goal). No reachable cell has f > 40. So 49 + 14 = 63 expansions is **unavoidable** on this map, whatever tie-breaking rule is used. It is a property of the map and the heuristic, not of the code.
- The A* program generated one extra frontier entry (65 vs 64). A cell's g was improved after it was first pushed, leaving one stale heap entry, which was skipped when popped.

**Extra experiment (different map, to show the difference clearly).** On an open warehouse with rows of shelves (`maps/open_warehouse.txt`):

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | yes | yes |
| Path length | 24 | 24 |
| States expanded | **71** | **36** |

Here A* expands about half as many states as BFS, while still finding a shortest path.

**(d) Why might A* expand fewer states?**
BFS is *blind*: it expands states in order of distance from the start, spreading out equally in every direction, including directly away from the goal. A* orders the frontier by f = g + h, so it uses **information about the goal** to expand first the states that appear to lie on a cheap path to it. States whose f exceeds the optimal cost are never expanded. The saving depends entirely on how well h reflects the real remaining cost. In the open warehouse, Manhattan distance is close to the true distance, so A* heads almost straight for the goal. In the lab maze, the true distance is often double the Manhattan distance, so the heuristic gives almost no useful guidance. A* then behaves like BFS, while paying the small extra overhead of a priority queue.

---

## 8. Task 6: Heuristic investigation

**Why is Manhattan distance appropriate?** (LLM explanation, then verified.) With only Up, Down, Left and Right moves of cost 1, any route from n to G needs at least |dx| horizontal moves and |dy| vertical moves, so h\*(n) ≥ |dx| + |dy|. Obstacles can only lengthen the route. So Manhattan distance never overestimates: it is **admissible**. It is also **consistent**, because one move changes it by at most 1, equal to the step cost. It equals the exact cost on an empty grid, so it is the most informed simple admissible heuristic for 4-way movement. The test `test_manhattan_admissible_on_lab_map` checks h(n) ≤ h\*(n) on every reachable cell.

### Results

**Laboratory warehouse** (optimal = 40)

| Heuristic | Solution found | Path length | States expanded | Optimal? |
|---|---|---|---|---|
| Manhattan | yes | 40 | 63 | yes |
| h(n) = 0 | yes | 40 | 63 | yes |
| Euclidean | yes | 40 | 63 | yes |
| 2 × Manhattan | yes | 40 | **67** | yes |

On the maze, every variant expands essentially the whole reachable space, for the reasons given in Task 5. So two extra maps were used to make the effect of the heuristic visible.

**Open warehouse** (optimal = 24)

| Heuristic | Solution found | Path length | States expanded | Optimal? |
|---|---|---|---|---|
| Manhattan | yes | 24 | 36 | yes |
| h(n) = 0 | yes | 24 | **71** | yes |
| Euclidean | yes | 24 | 60 | yes |
| 2 × Manhattan | yes | 24 | 36 | yes |

**Two-route map, Test 4** (optimal = 14)

| Heuristic | Solution found | Path length | States expanded | Optimal? |
|---|---|---|---|---|
| Manhattan | yes | 14 | 24 | yes |
| h(n) = 0 | yes | 14 | 27 | yes |
| Euclidean | yes | 14 | 24 | yes |
| 2 × Manhattan | yes | **16** | **16** | **NO** |

```
2 x Manhattan (16 moves)     Manhattan (14 moves)
#############                #############
#...........#                #***********#
#.#########.#                #*#########*#
#S********#G#                #S........#G#
#########*#*#                #########.#.#
#########*#*#                #########.#.#
#########***#                #########...#
#############                #############
```

### What the experiments show

**1. h(n) = 0.** A* becomes **uniform-cost search**. With unit step costs, that expands states in the same order as BFS. It always finds an optimal solution (h = 0 is trivially admissible) but uses no goal information: 71 expansions in the open warehouse, exactly the same as BFS, versus 36 with Manhattan.

**2. Euclidean distance.** Still **admissible** (a straight line is never longer than a 4-way path), so the solution is still optimal every time. But it is **less informed**: it is smaller than Manhattan whenever both dx and dy are non-zero, so it underestimates more. More states have f below the optimum and must be expanded: 60 in the open warehouse against Manhattan's 36. Euclidean is suited to free or diagonal movement, not to this robot.

**3. 2 × Manhattan.** **Not admissible.** Next to the goal, the true cost is 1 but h = 2. This is weighted A* with w = 2, which pushes the search greedily towards the goal:
- It can be **faster**: in the Test 4 map it expanded only 16 states (fewest of all).
- But it can return a **suboptimal path**: in the Test 4 map it followed the corridor that points directly at G and returned 16 moves instead of 14. The over-estimated h made the detour over the top (which first moves *away* from G) look too expensive to try.
- It can also **waste work**. On the lab maze it found 4 cells, (7,7) to (7,10) in the bottom dead-end corridor, by a worse route first, and had to re-expand them when cheaper routes appeared (67 expansions vs 63). This happens because the heuristic is inconsistent: h drops by 2 per step while the step costs only 1.
- It is not unboundedly bad. The randomised test confirms that over 500 maps its path was never more than 2 × the optimum, the theoretical bound for weighted A* with w = 2.

### Think About It: too optimistic vs too aggressive

From the experiments:
- **Too optimistic** (h too small, e.g. h = 0 or Euclidean): A* stays **correct and optimal**, but the guidance weakens and it degrades towards blind search, expanding more states. Admissibility guarantees optimality, but not efficiency. The closer an admissible h is to h\*, the fewer states are expanded.
- **Too aggressive** (h > h\*, e.g. 2 × Manhattan): A* behaves more like **greedy best-first search**. It often expands fewer states and finds *a* solution quickly, but it **loses the guarantee of optimality** (16 moves instead of 14). It may also redo work when it discovers better routes late.
- The ideal heuristic is as large as possible **while still never overestimating**. Manhattan distance is that heuristic for this robot. But even the ideal heuristic cannot help much when the environment (the lab maze) makes the straight-line estimate unrepresentative of the real distance.

---

## 9. Task 7: Evaluation of the LLM-generated agent

### What I designed, what the LLM suggested, what I accepted, changed and tested

| | |
|---|---|
| **Designed by me (before prompting)** | The problem formulation (Task 0); the state as a `(row, col)` tuple; the map as a list of strings; successor-function approach; goal test on removal from the frontier; frontier contents (f, g, state); parent-pointer path reconstruction; what to report. These were written into Prompt 1. |
| **Suggested by the LLM** | The `heapq` binary heap with `(f, h, counter, g, state)` entries; tie-breaking on h; the insertion counter (so states are never compared directly); the `best_g` dictionary plus stale-entry skipping ("lazy deletion"); a `SearchResult` dataclass; the command-line interface; the explanation of why Manhattan is admissible. |
| **Accepted** | The overall structure, the heap-based frontier, lazy deletion, the heuristic functions, path reconstruction. |
| **Changed** | (1) Required BFS to count "states expanded" in **exactly the same way** as A* (goal test on removal, goal not counted), instead of the textbook goal-test-on-generation, which would make the comparison unfair. (2) Made the heuristic a **parameter** so Task 6 does not need a separate program. (3) **Allowed a closed state to be reopened** if a cheaper path is found. This matters only for the inadmissible 2 × Manhattan heuristic, and without it the 4 re-expansions on the maze would have been hidden. (4) Fixed a wrong hard-coded expectation in the no-solution test (7 → 9 reachable cells, now computed rather than hard-coded). |
| **Tested** | All four required tests; a second "many equal paths" map; 500 random maps against an independent BFS; admissibility of Manhattan and Euclidean on every cell of the warehouse; inadmissibility of 2 × Manhattan; the 2× optimality bound for weighted A*. |

**1. What parts of the generated code were correct immediately?**
The core A* loop: the priority queue, the computation of g, h and f, the successor function, the goal test, and path reconstruction. These were correct on the first run. A* found the optimal 40-move path, and later tests confirmed it on all maps.

**2. Did you find any bugs or design problems?**
- **A bug in a test, not in the search.** The no-solution test asserted that 7 states would be expanded, but the map has 9 reachable cells. The program was right and the test's expectation was wrong.
- **A design problem in measurement.** The standard textbook BFS, and the BFS the LLM generated for Lab 1, performs the goal test when a state is *generated*, while A* must test when a state is *removed* from the frontier. Used unchanged, "states expanded" would mean different things for the two algorithms, and the Task 5 comparison would be misleading. Prompt 2 therefore specified a common counting convention, and the BFS was checked to follow it.
- **A subtle design choice with the inadmissible heuristic.** Without reopening closed states, A* with 2 × Manhattan silently ignores better paths found later. This is not wrong (weighted A* often does this deliberately), but it changes what the experiment measures, so it had to be a conscious decision.
- **A presentation bug.** The heuristic names contained `|` characters, which broke the Markdown result tables.

**3. How did you discover those problems?**
By **running tests with known expected answers**. The no-solution test failed (`9 != 7`), and counting the reachable cells by hand showed the expectation was wrong. The measurement issue was found by asking *"what exactly does 'states expanded' count?"* when planning the comparison and reading the code for Task 4. The surprising result that BFS and A* both expanded 63 states was investigated by computing how many cells have f < 40, rather than being dismissed as a bug or accepted without understanding.

**4. Did the LLM use terminology or data structures that you did not understand?**
At first: "lazy deletion" / "stale heap entries". Python's `heapq` has no decrease-key operation, so instead of updating an entry the code pushes a new, cheaper one and later skips the old one when it is popped. Also the tie-breaking counter, which is needed because Python would otherwise try to compare two states when f and h are equal. Both are now explained in comments in the code.

**5. Did you modify the LLM-generated code?**
Yes. See "Changed" above: a uniform counting convention, a pluggable heuristic, reopening of closed states, and test corrections.

**6. Which tests were most useful?**
- **Test 3 (no solution):** it proved the search terminates, and it exposed the wrong expectation.
- **Test 4 (alternative paths):** it is the only test that can distinguish a shortest path from merely *a* path. It later revealed the suboptimality of 2 × Manhattan.
- **The randomised comparison against an independent BFS** over 500 maps gives much stronger evidence of correctness than a handful of hand-made cases.

**7. Could you have trusted the program without testing it?**
No. The output on the lab map *looked* plausible, but a plausible path proves neither optimality nor correct failure handling nor correct counting. Without tests I would not have known the expansion counts were comparable, that 2 × Manhattan can return wrong (suboptimal) answers, or that the no-solution case terminates. Working output ≠ validated algorithm.

**8. What did you understand about A\* that you did not understand before implementing it?**
- A\* is only as good as its heuristic *for the particular environment*. On the maze, the admissible Manhattan heuristic gave no saving at all over BFS, because the true distance is twice the estimate.
- A\* must expand every state with f(n) < C\*. So the number of expansions can be predicted from the heuristic and the map, without running the search.
- The goal test must be on removal from the frontier, not on generation, for A\* to be optimal.
- Admissibility guarantees optimality, not efficiency. Overestimating trades optimality for speed.
- Practical implementation details such as tie-breaking, lazy deletion and reopening affect the measured numbers. They must be fixed and documented when comparing algorithms.

---

## 10. Final reflection

**1. Why is it important to formulate the search problem before writing the search algorithm?**
The search algorithm is generic. Everything specific to the problem lies in the formulation: what a state is, which actions are legal, what counts as reaching the goal, and what a move costs. Errors there cannot be fixed by a better algorithm. For example, if the state included unnecessary information, the state space would explode. If invalid moves were not excluded, the robot would drive through shelves. And if costs were not uniform, BFS would no longer be optimal. Writing the formulation first also gives the specification against which the code is tested, and makes it possible to write a precise prompt instead of a vague one. Generating code before the problem is understood just produces a program whose correctness cannot be judged.

**2. In what sense is A\* an "informed" search algorithm?**
A blind algorithm such as BFS uses only information from the problem definition and the path so far, g(n). It has no idea which direction the goal lies in. A\* also uses **domain knowledge about the goal**: the heuristic h(n) estimates the remaining cost from each state. By ordering the frontier on f = g + h, A\* expands first the states that appear to lie on a cheap complete path, and never expands states whose f exceeds the optimal cost. The "information" is the heuristic, here the knowledge that a robot restricted to grid moves needs at least |dx| + |dy| steps to reach the goal.

**3. Why does the choice of heuristic matter?**
The heuristic decides both the **efficiency** and the **correctness** of A\*. An admissible heuristic guarantees an optimal solution. Among admissible heuristics, the larger (better-informed) one expands fewer states: in the open warehouse, Manhattan expanded 36 states, Euclidean 60, and h = 0 71. An overestimating heuristic (2 × Manhattan) can be faster but returned a 16-move path where 14 was possible. And a heuristic that matches the geometry of the problem poorly gives almost no benefit, as the maze showed, where all heuristics expanded essentially every cell. The heuristic is where knowledge about the domain enters the algorithm, so choosing it is an engineering decision that must be justified and tested.

**4. What did the LLM contribute to the engineering process?**
It translated the design into clean, working Python very quickly. It suggested good practical techniques that I would not necessarily have known: a heap with a tie-breaking counter, lazy deletion of stale entries, and a result dataclass. It explained why Manhattan distance is admissible, generated the BFS variant and the pluggable heuristics, produced a first set of tests, and helped diagnose the failing test. It acted like a fast, knowledgeable junior colleague. It reduced the effort of writing code, so more time could go into design, testing and analysing results.

**5. What could go wrong if an engineer simply accepted LLM-generated code without testing it?**
The code could produce plausible but wrong output, and nobody would notice. In this lab:
- the expansion counts of BFS and A\* would have been measured inconsistently, giving a misleading comparison;
- a wrong test expectation would have gone unexamined;
- the fact that a "faster" heuristic returns non-optimal routes would have gone unnoticed.

In a real warehouse, unvalidated code might send robots on inefficient routes, fail to report that a destination is unreachable, loop forever on an unusual map, or even (if obstacle checking were wrong) drive into shelves. LLM output is a *hypothesis* about a correct implementation. The engineer remains responsible for understanding it, testing it against cases with known answers, and validating it before relying on it.
