% =====================================================================
% road.pl  -  Task 8: connecting Prolog to logical reasoning
%
% Run:  swipl -q -g "(reduce_speed -> writeln(true) ; writeln(false))" -t halt road.pl
% =====================================================================

% Fact
wet_road.

% Rule:  WetRoad -> Slippery
slippery :-
    wet_road.

% Rule:  Slippery -> ReduceSpeed
reduce_speed :-
    slippery.
