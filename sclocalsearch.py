import numpy as np
import os
import time
import numba
from numba import njit

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


@njit
# @jit
def build_new_instance(instance, rowcounts, removed_columns, start_solution):
    # find rows uncovered by the removed columns and update rowcounts
    uncovered_rows = [0]
    # uncovered_rows = numba.typed.empty_list(numba.i8)

    new_rowcounts = rowcounts.copy()

    for c_idx in removed_columns:
        for r_idx in instance.matrix[c_idx]:
            if r_idx == -1:
                break
            if new_rowcounts[r_idx] == 1:
                uncovered_rows.append(r_idx)
            new_rowcounts[r_idx] -= 1

    assert (new_rowcounts >= 0).all()
    # del uncovered_rows[0]
    uncovered_rows = uncovered_rows[1:]

    # Derive the set of new columns to explore
    # (np.setdiff1d is not supported by numba)
    new_start_sol = [el for el in start_solution if el not in removed_columns]
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


def scLocalSearch(
    instance: Instance,
    start_time,
    n_size=5,
    time_limit=10,
    max_patience=500,
    seed=0,
):
    # Find a feasible solution using a greedy algorithm
    print("--- Finding a feasible solution ---")
    incumbent, best_cost, rowcounts = scGreedy(
        instance,
        in_rowcounts=np.zeros(instance.m, dtype=np.int32),
        start_solution=numba.typed.List([0]),
    )
    current_solution = incumbent.copy()
    current_rowcounts = rowcounts.copy()
    current_cost = best_cost

    end_time = time.time()
    t = end_time - start_time
    print("Feasible solution of value: {} [time {:.2f}]".format(best_cost, t))

    # define numpy rng
    rng = np.random.default_rng(seed=seed)

    step = 0
    patience = 0

    while True:
        step += 1
        patience += 1
        if step % 100 == 0:
            print("[Local Search] Step {}".format(step))
        end_time = time.time()
        t = end_time - start_time
        if t > time_limit:
            print(
                f"Time limit [{time_limit}s] reached [{t:.3f}s passed], stopping search..."
            )
            break

        # Remove n_size random columns from the solution
        removed_columns = rng.choice(current_solution, size=n_size, replace=False)

        new_instance, new_rowcounts, new_start_sol = build_new_instance(
            instance, current_rowcounts, removed_columns, current_solution
        )

        local_sol, local_cost, local_rowcounts = scGreedy(
            new_instance,
            in_rowcounts=new_rowcounts,
            start_solution=[0] + new_start_sol,
        )

        if local_cost < current_cost:
            print("Found better solution")
            patience = 0
            current_cost = local_cost
            current_solution = local_sol
            current_rowcounts = local_rowcounts

            if local_cost < best_cost:
                best_cost = local_cost
                incumbent = local_sol
                rowcounts = local_rowcounts

                end_time = time.time()
                t = end_time - start_time

                print(
                    f"Feasible solution of value {best_cost} [time {t:.2f}]]".format(
                        best_cost, t
                    )
                )

        if patience > max_patience:
            patience = 0
            print(f"Patience limit [{max_patience}] reached, perturbating solution...")
            # Perturbate the solution
            removed_columns = rng.choice(incumbent, size=n_size * 2, replace=False)

            new_instance, new_rowcounts, new_start_sol = build_new_instance(
                instance, rowcounts, removed_columns, incumbent
            )

            current_solution, current_cost, current_rowcounts = scGreedy(
                new_instance,
                in_rowcounts=new_rowcounts,
                start_solution=[0] + new_start_sol,
            )

            end_time = time.time()
            t = end_time - start_time
            print(f"Perturbed solution of cost {current_cost} [time {t:.2f}]]")

    return incumbent, best_cost, rowcounts
