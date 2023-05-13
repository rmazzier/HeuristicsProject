import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import os
import time
from numba import jit

from utils import (
    import_instance,
    Instance,
    transpose_column_major,
    compute_cost,
    solution_is_valid,
    all_instances,
)

from scgreedy import scGreedy
from scgrasp import scGrasp
from utils import save_solution_to_file


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
    # solution, rowcounts, start_time = scGrasp(instance)
    num_steps = 0

    while num_steps < 10:
        newsols = []
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
                newsol_cost = compute_cost(newsol, instance)
                if newsol_cost < compute_cost(solution, instance):
                    newsols.append(
                        (column_to_remove, column_to_add, newsol, newsol_cost)
                    )

        if len(newsols) == 0:
            print(
                f"No better solution found, tried other {n_alternate_solutions_tried}"
            )
            save_solution_to_file(
                solution,
                instance,
                f"./solutions/localsearch/{instance.name}.0.sol",
            )
            return solution

        # find the best solution among the newsols
        costs = [s[3] for s in newsols]
        best_idx = np.argmin(costs)

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
        solution = newsols[best_idx][2]


@jit(nopython=False)
def method(rowcounts, removed_columns, best_solution):
    # find rows uncovered by the removed columns and update rowcounts
    uncovered_rows = []
    new_rowcounts = rowcounts.copy()

    for c_idx in removed_columns:
        for r_idx in instance.matrix[c_idx]:
            if r_idx == -1:
                break
            if new_rowcounts[r_idx] == 1:
                uncovered_rows.append(r_idx)
            new_rowcounts[r_idx] -= 1

    assert (new_rowcounts >= 0).all()

    # # Take unique values
    # uncovered_rows = np.unique(uncovered_rows)

    # Derive the set of new columns to explore
    # new_start_sol = np.setdiff1d(best_solution, removed_columns)
    new_start_sol = [el for el in best_solution if el not in removed_columns]
    # new_columns = np.setdiff1d(np.array(range(instance.n)), new_start_sol)
    new_columns = [el for el in range(instance.n) if el not in new_start_sol]

    # Update the covered_rows vector
    new_covered_rows = np.zeros(instance.n).astype("int64")

    for c_idx in new_columns:
        for r_idx in instance.matrix[c_idx]:
            if r_idx == -1:
                break
            if r_idx in uncovered_rows:
                new_covered_rows[c_idx] += 1
    assert (new_covered_rows >= 0).all()
    # print(new_covered_rows.nonzero()[0].shape[0])

    # Now we solve a smaller instance of SCP only on the uncovered rows, using the columns not in the solution
    new_instance = Instance(
        instance.m,
        instance.n,
        instance.costs,
        new_covered_rows.copy(),
        instance.matrix,
        instance.transposed_matrix,
        instance.name,
    )

    return new_instance, new_rowcounts, new_start_sol


def scLocalSearch2(
    instance: Instance,
    start_time,
    n_size=5,
    time_limit=10,
    seed=0,
):
    # Find a feasible solution using a greedy algorithm
    print("--- Finding a feasible solution ---")
    best_solution, best_cost, rowcounts = scGreedy(
        instance,
        in_rowcounts=np.zeros(instance.m, dtype=np.int32),
        start_solution=[0],
    )
    end_time = time.time()
    t = end_time - start_time
    print("Feasible solution of value: {} [time {:.2f}]".format(best_cost, t))

    # define numpy rng
    rng = np.random.default_rng(seed=seed)

    step = 0

    while True:
        step += 1
        print("[Local Search] Step {}".format(step))
        end_time = time.time()
        t = end_time - start_time
        if t > time_limit:
            print(
                f"Time limit [{time_limit}s] reached [{t:.3f}s passed], stopping search..."
            )
            break

        # Remove n_size random columns from the solution
        removed_columns = rng.choice(best_solution, size=n_size, replace=False)

        new_instance, new_rowcounts, new_start_sol = method(
            rowcounts, removed_columns, best_solution
        )

        local_sol, local_cost, local_rowcounts = scGreedy(
            new_instance,
            in_rowcounts=new_rowcounts,
            start_solution=[0] + new_start_sol,
        )

        if local_cost < best_cost:
            best_cost = local_cost
            best_solution = local_sol
            rowcounts = local_rowcounts

            end_time = time.time()
            t = end_time - start_time

            print(
                "Found better solution of cost {} [time {:.2f}]]".format(best_cost, t)
            )
    return best_solution, best_cost, rowcounts


if __name__ == "__main__":

    paths = [
        os.path.join("rail", "instances", "rail507"),
        os.path.join("rail", "instances", "rail516"),
        os.path.join("rail", "instances", "rail582"),
        os.path.join("rail", "instances", "rail2536"),
        os.path.join("rail", "instances", "rail2586"),
        os.path.join("rail", "instances", "rail4284"),
        os.path.join("rail", "instances", "rail4872"),
    ]
    for instance_path in paths:

        start_time = time.time()
        instance = import_instance(instance_path)
        solution, cost, _ = scLocalSearch2(
            instance,
            n_size=40,
            start_time=start_time,
            time_limit=600,
            seed=0,
        )

        # print(f"--- {instance.name} ---")

        # solution, best_cost, rowcounts = scGreedy(
        #     instance,
        #     in_rowcounts=np.zeros(instance.m, dtype=np.int32),
        #     start_solution=[0],
        # )

        # print(best_cost)

        save_solution_to_file(
            solution,
            instance,
            f"./solutions/localsearch/{instance.name}.0.sol",
        )
