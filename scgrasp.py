import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import os
import time

from utils import (
    import_instance,
    Instance,
    transpose_column_major,
    compute_cost,
    solution_is_valid,
)

from scgreedy import scGreedy


def one_opt_neighborhood(col_idx, solution, instance: Instance, rowcounts):
    """Returns indices of columns that can replace the column of index col_idx in the solution"""

    # First check if the solution is valid
    assert solution_is_valid(solution, instance)
    assert col_idx in solution
    assert not 0 in rowcounts

    rowcounts = rowcounts.copy()
    # First, find the rows that are covered by the column of index col_idx
    # and subtract 1 from their rowcounts
    # If, after that, I find a row that has a rowcount of 0, then its added to the list of uncovered rows
    uncovered = []
    for el in instance.matrix[col_idx]:
        if el == -1:
            break
        rowcounts[el] -= 1
        if rowcounts[el] == 0:
            uncovered.append(el)

    # Now, for each uncovered row, I get all the columns that cover it
    covering_columns_per_row = []
    for r_idx in uncovered:
        covering_columns = []
        for c_idx in instance.transposed_matrix[r_idx]:
            if c_idx == -1:
                break
            covering_columns.append(c_idx)
        covering_columns_per_row.append(covering_columns)

    if len(covering_columns_per_row) == 0:
        return []

    substitutes = reduce(np.intersect1d, covering_columns_per_row)
    return substitutes


def scLocalSearch(instance: Instance):
    print("--- Start Local Search ---")
    # first find a feasible solution using the greedy algorithm
    solution, rowcounts, start_time = scGreedy(instance)
    newsols = []
    num_steps = 0
    while num_steps < 10:
        # assert solution_is_valid(solution, instance)
        n_alternate_solutions_tried = 0
        num_steps += 1
        print(f"Step {num_steps}")
        for column_to_remove in solution:
            substitutes = one_opt_neighborhood(
                column_to_remove, solution, instance, rowcounts
            )
            for column_to_add in substitutes:
                n_alternate_solutions_tried += 1
                newsol = solution.copy()
                newsol.remove(column_to_remove)
                newsol.append(column_to_add)
                assert solution_is_valid(newsol, instance)
                # print(compute_cost(newsol, instance))
                if compute_cost(newsol, instance) < compute_cost(solution, instance):
                    newsols.append((column_to_remove, column_to_add, newsol))

        if len(newsols) == 0:
            print(
                f"No better solution found, tried other {n_alternate_solutions_tried}"
            )
            return solution

        # find the best solution among the newsols
        costs = [compute_cost(s[2], instance) for s in newsols]
        best_idx = np.argmin(costs)

        # TODO: figure out what's wrong with this (I get an error where somewhere rowcounts is 0)

        # update rowcounts:
        # first subtract one to the rowcounts of the rows covered by the column that was removed
        for j in instance.matrix[newsols[best_idx][0]]:
            if j != -1:
                rowcounts[j] -= 1
            else:
                break
        # then add one from the rowcounts of the rows covered by the column that was added
        for j in instance.matrix[newsols[best_idx][1]]:
            if j != -1:
                rowcounts[j] += 1
            else:
                break

        assert solution_is_valid(newsols[best_idx][2], instance)
        end = time.time()
        t = end - start_time
        print(f"Feasible solution of value: {costs[best_idx]} [time {t:.2f}]")
        solution = newsols[best_idx][-1]


if __name__ == "__main__":
    instance_path = os.path.join("rail", "instances", "rail582")
    instance = import_instance(instance_path)
    solution = scLocalSearch(instance)
