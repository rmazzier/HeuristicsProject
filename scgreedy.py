import numpy as np
import os
import time

from utils import import_instance, Instance, transpose_column_major, compute_cost


def scgreedy(instance: Instance):
    """Solve the set covering problem on a given instance using the greedy algorithm"""

    # at each iteration, i must select the column that maximizes the ratio covered_rows/cost

    print("--- Start scgreedy algorithm ---")
    # Get the transposed column-major matrix
    transposed_matrix = transpose_column_major(instance.matrix, instance.m)

    # list of zero-based indices
    solution = []

    # rowcounts stores the amount of times each row is covered by the current solution
    rowcounts = np.zeros(instance.m, dtype=np.int32)
    # colcounts stores the amount of covered rows by each column, among those rows that are still uncovered by the current solution
    colcounts = instance.covered_rows

    while True:
        best_score = 0
        best_i = -1
        for i in range(instance.n):
            score = colcounts[i] / instance.costs[i]
            assert score >= 0
            if score > best_score:
                # print("New best score: {} (column {})".format(score, i))
                best_score = score
                best_i = i

        # Check wether no more rows can be covered, and if so, break
        if best_i == -1:
            print(
                f"\nAlgorithm completed. {rowcounts.nonzero()[0].shape[0]}/{instance.m} covered rows."
            )
            cost = compute_cost(solution, instance)
            print(f"Total cost: {cost}")
            break

        # print("Adding column {}".format(best_i))
        solution.append(best_i)
        # print("Current solution: {}".format(solution))

        # Now update rowcounts and colcounts
        bestcol = instance.matrix[best_i]

        # now, for each row that has just been covered by best_i, look for all the columns that would cover such row, and decrease their colcount by 1,
        # but only if that row was not already covered before.
        for r in bestcol:
            if r == -1:
                break
            if rowcounts[r] == 0:
                relative_columns = transposed_matrix[r]
                for rc in relative_columns:
                    if rc == -1:
                        break
                    # if colcounts[rc] > 0:
                    colcounts[rc] -= 1
                    assert colcounts[rc] >= 0

            rowcounts[r] += 1

        print(
            f"Completion: {(rowcounts.nonzero()[0].shape[0] / instance.m) * 100:.3f}% ",
            end="\r",
        )

    return solution


if __name__ == "__main__":
    # instance_path = os.path.join("rail", "instances", "rail507")
    # instance_path = os.path.join("rail", "instances", "rail516")
    # instance_path = os.path.join("rail", "instances", "rail582")
    instance_path = os.path.join("rail", "instances", "rail2536")
    instance = import_instance(instance_path)
    solution = scgreedy(instance)
