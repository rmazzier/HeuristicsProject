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
    alpha,
    time_limit,
    max_patience,
    seed=0,
):
    """
    Solve the set covering problem on a given instance using the local search algorithm with perturbation

    Parameters
    ----------
    instance : Instance
        The instance to solve
    start_time : float
        The time at which the algorithm started
    alpha : int
        The number of columns to remove from the solution at each perturbation step
    time_limit : int
        The maximum time allowed for the algorithm to run
    max_patience : int
        The maximum number of iterations without improvement before perturbating the solution
    seed : int, optional
        The seed for the random number generator (default is 0)

    """
    print("--- Finding a feasible solution ---")
    incumbent, best_cost, rowcounts = scGreedy(
        instance,
        in_rowcounts=np.zeros(instance.m, dtype=np.int32),
        start_solution=[0],  # quick and dirty way to tell numba the type of the list
    )
    current_solution = incumbent.copy()
    current_rowcounts = rowcounts.copy()
    current_cost = best_cost

    end_time = time.time()
    t = end_time - start_time
    print("Feasible solution of value: {} [time {:.2f}]".format(best_cost, t))

    # define numpy rng to allow for reproducibility
    rng = np.random.default_rng(seed=seed)

    step = 0
    patience = 0
    sol_idx = 0

    while True:
        step += 1
        patience += 1
        # if step % 100 == 0:
        #     print("[Local Search] Step {}".format(step))
        end_time = time.time()
        t = end_time - start_time
        if t > time_limit:
            print(
                f"Time limit [{time_limit}s] reached [{t:.3f}s passed], stopping search..."
            )
            break

        # Remove n_size random columns from the solution
        removed_columns = rng.choice(current_solution, size=alpha, replace=False)

        # Build a new instance of the problem with the removed columns
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

                sol_idx += 1

                # create the directory if it does not exist
                out_dir = os.path.join("solutions", instance.name)

                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(
                    out_dir,
                    f"{instance.name}.{sol_idx}.sol",
                )

                save_solution_to_file(
                    incumbent,
                    instance,
                    out_path,
                )

        if patience > max_patience:
            patience = 0
            print(f"Patience limit [{max_patience}] reached, perturbating solution...")

            # Perturbate the solution
            removed_columns = rng.choice(incumbent, size=alpha * 2, replace=False)

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

    return incumbent, best_cost
