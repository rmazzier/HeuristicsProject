import numpy as np
import numba
from numba import njit, jit, int64
from numba.types import string
from numba.experimental import jitclass

# from solchecker import readInstance

all_instances = [
    "rail507",
    "rail516",
    "rail582",
    "rail2536",
    "rail2586",
    "rail4284",
    "rail4872",
]

spec = [
    ("m", int64),
    ("n", int64),
    ("costs", int64[:]),
    ("covered_rows", int64[:]),
    ("matrix", int64[:, :]),
    ("transposed_matrix", int64[:, :]),
    ("name", string),
]


@jitclass(spec)
class Instance:
    def __init__(self, m, n, costs, covered_rows, matrix, transposed_matrix, name=None):
        """Represents a rail instance"""
        self.name = name
        self.m = m
        self.n = n
        self.costs = costs
        self.covered_rows = covered_rows
        # convert to 0-indexing
        self.matrix = matrix
        self.transposed_matrix = transposed_matrix


def import_instance(instance_path):
    """Read rail* instance from column-major format and returns and Instance object

    NB: During parsing, the 1-based indices of the source files are converted to 0-based indices"""

    print(f"Reading instance from {instance_path}")

    instance_array = []
    with open(instance_path) as fp:
        lines = fp.readlines()
        for line in lines:
            instance_array.append([int(i) for i in line[1:-2].split(" ")])

    m = instance_array[0][0]
    n = instance_array[0][1]
    matrix = instance_array[1:]

    # Convert our column-major list of lists to a numpy array for faster performance
    maxlen = max([len(list) for list in matrix])
    padded_matrix = []

    for i in matrix:
        # pad with zeros each list to the same length = maxlen
        padded_matrix.append(i + [0] * (maxlen - len(i)))

    matrix = np.array(padded_matrix)

    # preprocessing before passing to Instance class
    costs = matrix[:, 0]
    covered_rows = matrix[:, 1]
    # convert to 0-indexing
    matrix = matrix[:, 2:] - 1
    t_matrix = transpose_column_major(matrix, m)
    return Instance(
        m, n, costs, covered_rows, matrix, t_matrix, instance_path.split("/")[-1]
    )


@njit
def transpose_column_major(matrix, m):
    """NB: assumes input matrix is already with zero-index"""

    # print("Transposing matrix...")

    # Define list of typed empty lists
    # transposed = [numba.typed.List.empty_list(numba.i8) for _ in range(m)]
    transposed = [[0] for _ in range(m)]

    for i, row in enumerate(matrix):
        for element in row:
            if element != -1:
                transposed[element].append(i)

    # We have array of lists of uneven length, so we need to pad it
    maxlen = max([len(list) for list in transposed])

    for i in transposed:
        n_ones = maxlen - len(i)
        for _ in range(n_ones):
            i.append(-1)

        # Remove the dummy element
        # i = i[1:]
        del i[0]

    return np.array(transposed)


# def transpose_column_major(matrix, m):
#     """NB: assumes input matrix is already with zero-index"""

#     # print("Transposing matrix...")
#     transposed = [[] for _ in range(m)]
#     # transposed = np.empty(m, dtype=np.int64)
#     for i, row in enumerate(matrix):
#         for element in row:
#             if element != -1:
#                 transposed[element].append(i)
#             pass

#     # pad each list to the same length
#     maxlen = max([len(list) for list in transposed])
#     padded_matrix = []

#     for i in transposed:
#         # pad with -1s each list to the same length = maxlen
#         padded_matrix.append(i + [-1] * (maxlen - len(i)))

#     return np.array(padded_matrix)


@njit
def compute_cost(solution, instance: Instance):
    cost = 0
    for s in solution:
        cost += instance.costs[s]
    return cost


@njit
def solution_is_valid(solution, instance):
    covered = np.zeros(instance.m)
    for c_idx in solution:
        for r_idx in instance.matrix[c_idx]:
            if r_idx == -1:
                break
            covered[r_idx] = 1

    return covered.all()


def save_solution_to_file(solution, instance, filename):
    with open(filename, "w") as fp:
        fp.write(f"{compute_cost(solution, instance)}")
        fp.write("\n")
        for s in solution:
            fp.write(str(s))
            fp.write(" ")


if __name__ == "__main__":
    pass
