% =====================================================================
% planner.pl  -  Prolog as a logical verifier for the warehouse planner
% Laboratory: Logical Reasoning for Planning, Optional Extension (Tasks 6, 7)
%
% Load in SWI-Prolog:   swipl planner.pl
% Run all lab queries:  swipl -q -g run_all -t halt planner.pl
% =====================================================================

% ---------------------------------------------------------------------
% Task 6: facts describing the warehouse
% ---------------------------------------------------------------------
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% Rule:  Connected(X,Y) -> CanMove(X,Y)
can_move(X,Y) :-
    connected(X,Y).

% ---------------------------------------------------------------------
% Task 7: checking a proposed plan
% ---------------------------------------------------------------------
valid_move(X,Y) :-
    connected(X,Y).

% ---------------------------------------------------------------------
% Beyond the sheet: verifying a whole movement plan.
% A plan is a list of move(X,Y) terms.  It is valid from location Start
% if each move starts where the robot currently is and is a valid_move.
% ---------------------------------------------------------------------
valid_route(_, []).
valid_route(Here, [move(Here,Next) | Rest]) :-
    valid_move(Here, Next),
    valid_route(Next, Rest).

% ---------------------------------------------------------------------
% Beyond the sheet: verifying a complete plan with PickUp and Drop.
% State = state(RobotLocation, PackageLocation) where PackageLocation is
% a location or the atom held.  Preconditions and effects follow the
% lab sheet exactly.
% ---------------------------------------------------------------------

% step(+Action, +StateBefore, -StateAfter): only succeeds when the
% action's preconditions hold in StateBefore.
step(move(X,Y),   state(X,P),    state(Y,P))    :- valid_move(X,Y).
step(pickup(L),   state(L,L),    state(L,held)).          % At(Robot,L), At(Package,L)
step(drop(L),     state(L,held), state(L,L)).             % At(Robot,L), Holding(Package)

% valid_plan(+Plan, +Initial, -Final)
valid_plan([], S, S).
valid_plan([A|As], S0, S) :-
    step(A, S0, S1),
    valid_plan(As, S1, S).

% The goal of the lab: At(Package, c)
goal(state(_, c)).

achieves_goal(Plan) :-
    valid_plan(Plan, state(a,a), Final),
    goal(Final).

% ---------------------------------------------------------------------
% Helper: run every query from the lab sheet and print the answers.
% ---------------------------------------------------------------------
show(Label, Goal) :-
    (   call(Goal) -> R = true ; R = false ),
    format("?- ~w.~n   ~w~n", [Label, R]).

run_all :-
    format("--- Task 6 ---~n"),
    show('can_move(a,b)', can_move(a,b)),
    show('can_move(a,c)', can_move(a,c)),
    format("~n--- Task 7 ---~n"),
    show('valid_move(a,b)', valid_move(a,b)),
    show('valid_move(b,c)', valid_move(b,c)),
    show('valid_move(a,c)', valid_move(a,c)),
    format("~n--- Task 7 challenge: proposed Move(a,c) ---~n"),
    show('valid_move(a,c)', valid_move(a,c)),
    show('valid_route(a,[move(a,b),move(b,c)])',
         valid_route(a,[move(a,b),move(b,c)])),
    show('valid_route(a,[move(a,c)])', valid_route(a,[move(a,c)])),
    format("~n--- Extension: checking complete plans ---~n"),
    show('achieves_goal([pickup(a),move(a,b),move(b,c),drop(c)])',
         achieves_goal([pickup(a),move(a,b),move(b,c),drop(c)])),
    show('achieves_goal([move(a,b),pickup(b),move(b,c),drop(c)])',
         achieves_goal([move(a,b),pickup(b),move(b,c),drop(c)])),
    show('achieves_goal([move(a,b),move(b,c)])',
         achieves_goal([move(a,b),move(b,c)])),
    show('achieves_goal([pickup(a),move(a,c),drop(c)])',
         achieves_goal([pickup(a),move(a,c),drop(c)])),
    format("~n--- All places reachable in one move from b ---~n"),
    findall(Y, can_move(b,Y), Ys),
    format("?- findall(Y, can_move(b,Y), Ys).~n   Ys = ~w~n", [Ys]),
    format("~n--- Cross-check: shortest complete plan found by Prolog ---~n"),
    shortest_plan(P),
    format("?- shortest_plan(P).~n   P = ~w~n", [P]).

% Iterative deepening: try plans of length 0, 1, 2, ... (max 8)
shortest_plan(Plan) :-
    between(0, 8, N),
    length(Plan, N),
    achieves_goal(Plan), !.
