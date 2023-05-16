import numpy as np
import os
import time


from utils import (
    import_instance,
    Instance,
    transpose_column_major,
    compute_cost,
    solution_is_valid,
    save_solution_to_file,
)


def scGrasp(instance: Instance, in_rowcounts, start_solution, alpha=0.95, seed=0):
    """Solve the set covering problem on a given instance using the greedy with GRASP"""

    # define numpy rng
    rng = np.random.default_rng(seed=seed)

    # print("--- Start scGrasp algorithm ---")
    start = time.time()

    # list of zero-based indices
    solution = start_solution

    # rowcounts stores the amount of times each row is covered by the current solution
    rowcounts = in_rowcounts.copy()
    # else:
    #     rowcounts = np.zeros(instance.m, dtype=np.int32)

    # colcounts stores the amount of covered rows by each column, among those rows that are still uncovered by the current solution
    colcounts = instance.covered_rows.copy()

    d = []
    e = []

    while True:
        completion = rowcounts.nonzero()[0].shape[0] / instance.m

        best_score = 0
        best_i = -1
        all_scores = []

        for i in range(instance.n):
            score = colcounts[i] / instance.costs[i]
            # score = colcounts[i] / instance.costs[i] ** 2
            assert score >= 0
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_i = i

        # Check wether no more rows can be covered, and if so, break
        if best_i == -1:
            assert solution_is_valid(solution, instance)
            cost = compute_cost(solution, instance)
            end = time.time()
            t = end - start

            # print(f"Feasible solution of value: {cost} [time {t:.2f}]")
            break

        # max_idxs = np.argwhere(np.array(all_scores) == best_score).flatten()

        sub_optimal_idxs = [
            i
            for i in range(len(all_scores))
            if (all_scores[i] >= best_score - 1) and (all_scores[i] > 0)
        ]

        # alpha = min(((completion) * 2 + 0.5), 0.95)
        e.append(alpha)
        if rng.random() > alpha:
            best_i = rng.choice(sub_optimal_idxs)

        d.append(all_scores[best_i])
        # print(best_i)

        solution.append(best_i)

        # Now update rowcounts and colcounts
        bestcol = instance.matrix[best_i]

        # now, for each row that has just been covered by best_i, look for all the columns that would cover such row, and decrease their colcount by 1,
        # but only if that row was not already covered before.
        for r in bestcol:
            # if r == 377:
            #     print("\n377 Found")
            if r == -1:
                break
            if rowcounts[r] == 0:
                relative_columns = instance.transposed_matrix[r]
                for rc in relative_columns:
                    if rc == -1:
                        break
                    # if colcounts[rc] > 0:
                    colcounts[rc] -= 1
                    assert colcounts[rc] >= 0

            rowcounts[r] += 1

        print(
            f"Completion: {completion * 100:.3f}% ",
            end="\r",
        )

    return solution, compute_cost(solution, instance), rowcounts, start
