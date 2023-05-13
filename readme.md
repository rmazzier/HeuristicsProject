# Heuristics for Mathematical Optimization 22/23

## Project description
Final project for Heuristics for Mathematical Optimization PhD Course, for which I developed a simple heuristc to find feasible (and good) solutions the [following rail instances](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html).

 - rail507
 - rail516
 - rail582
 - rail2536
 - rail2586
 - rail4284
 - rail4872

Those are instances of [**crew scheduling problem**](https://en.wikipedia.org/wiki/Crew_scheduling) (an example of Set Covering Problem), where a set of train/plane/whatever solutions must be entirely covered by a subset of pilots/crew members, under some costrains and costs which are encoded in a sparse matrix form (column-major format). Quality of the solution is determined by the sum of the costs of its elements, which we aim to minimize.



 ## Heuristic Description
 This heuristic executes a local search of the solution space, where the incumbent (current best solution) is perturbed if no better solution is found after a certain number of iterations. 

 The neighborhood of radius $r$ of $S$, denoted with $N(S,r)$, is defined by all the feasible solutions found by the Greedy algorithm initialized on $S \setminus R, R \subset S, |R|=r$.


 The hyperparameters of the algorithm are:
 - `alpha`: the size of the neighborhood;
 - `patience`: The number of iterations that the algorithm is allowed to search for better solutions in the current neighborhood, before perturbating the incumbent.

### Outline of the algorithm
 
 1) A feasible solution $S$ is found by a simple Greedy algorithm. At this step, $S$ is also the incumbent, $S^*$ (the best solution found yet).

 2) The neighborhood is randomly searched, as its dimension is too big to be explored exhaustively. Solutions $S' \in N(S,$ `alpha`$)$ are uniformely sampled until a new solution is found such that $C(S') < C(S)$. If so, $S= S'$. If $C(S') < C(S^*)$, then $S = S' = S^*$.
 3) If after `patience` iterations $C(S') > C(S)$, the algorithm picks a random feasible solution $S \in N(S^*, 2$ `alpha`$)$ as the center of the next neighborhood to search, and goes back to step 2.

The algorithm terminates after a given time budget, which was set to 10 minutes for this project.

## Implementation
The algorithm was implemented in Python. To boost the code performance, I used [Numba](https://numba.pydata.org/), a powerful python package that allows to translate Python functions to optimized machine code at runtime.

## Requirements
To run the project simply install the following Python packages
```
numpy==1.22.4
numba==0.55.2
```
 
 ## Results
 The following results were obtained by running the algorithm for 10 minutes with a 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz CPU.
| Instance | Greedy | Best Known | Local Search (Mine)  |   Alpha       |   Patience  | 
|:--------:|:------:|:----------:|:----------:          |:----------:   |:----------: | 
|  rail507 |   216  |     174    |     191              |      50         |       1000      | 
|  rail516 |   204  |     182    |     185              |      50         |       1000     | 
|  rail582 |   251  |     211    |     229              |      50         |       1000      | 
| rail2536 |   894  |     689    |     829              |      50         |       200     | 
| rail2586 |  1166  |     960    |     1101             |      50         |       200      | 
| rail4284 |  1376  |    1077    |     1295             |      50         |       200      | 
| rail4872 |  1902  |    1556    |     1814             |      50         |       200      | 

All the experiments can be reproduced with the exact parameters of the table by running the `run_heuristic.py` file.