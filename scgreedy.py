from numba import njit

from utils import (
    Instance,
    compute_cost,
    solution_is_valid,
)


@njit
def scGreedy(instance: Instance, in_rowcounts, start_solution):
    """Solve the set covering problem on a given instance using the greedy algorithm"""

    # list of zero-based indices
    solution = start_solution

    # rowcounts stores the amount of times each row is covered by the current solution
    rowcounts = in_rowcounts.copy()
    # colcounts stores the amount of covered rows by each column, among those rows that are still uncovered by the current solution
    colcounts = instance.covered_rows

    while True:

        best_score = 0
        best_i = -1
        for i in range(instance.n):
            score = colcounts[i] / instance.costs[i]
            assert score >= 0
            if score > best_score:
                best_score = score
                best_i = i

        # Check wether no more rows can be covered, and if so, break
        if best_i == -1:
            assert solution_is_valid(solution, instance)
            break

        solution.append(best_i)

        # Now update rowcounts and colcounts
        bestcol = instance.matrix[best_i]

        # now, for each row that has just been covered by best_i, look for all the columns that would cover such row, and decrease their colcount by 1,
        # but only if that row was not already covered before.
        for r in bestcol:
            if r == -1:
                break
            if rowcounts[r] == 0:
                relative_columns = instance.transposed_matrix[r]
                for rc in relative_columns:
                    if rc == -1:
                        break
                    colcounts[rc] -= 1
                    assert colcounts[rc] >= 0

            rowcounts[r] += 1

    # remove the dummy element we introduced for numba as we don't need it anymore
    del solution[0]

    return solution, compute_cost(solution, instance), rowcounts


if __name__ == "__main__":
    pass
