import numpy as np
import matplotlib.pyplot as plt


class Instance:
    def __init__(self, m, n, matrix):
        """Represents a rail instance"""
        self.m = m
        self.n = n
        self.costs = matrix[:, 0]
        self.covered_rows = matrix[:, 1]
        # convert to 0-indexing
        self.matrix = matrix[:, 2:] - 1


def import_instance(instance_path):
    """Read rail* instance from column-major format and returns:

    m: number of rows
    n: number of columns
    matrix: the column major formatted data, padded with zeros to make each entry of the same length

    NB: Remember that the indices reported in this matrix are 1-based"""

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
    return Instance(m, n, matrix)


def transpose_column_major(matrix, m):
    """NB: assumes input matrix is already with zero-index"""

    print("Transposing matrix...")
    transposed = [[] for _ in range(m)]
    for i, row in enumerate(matrix):
        for element in row:
            if element != -1:
                transposed[element].append(i)
            pass

    # pad each list to the same length
    maxlen = max([len(list) for list in transposed])
    padded_matrix = []

    for i in transposed:
        # pad with -1s each list to the same length = maxlen
        padded_matrix.append(i + [-1] * (maxlen - len(i)))

    return np.array(padded_matrix)


def compute_cost(solution, instance: Instance):
    cost = 0
    for s in solution:
        cost += instance.costs[s]
    return cost


if __name__ == "__main__":
    instance = import_instance("./rail/instances/rail4872")
    print(instance.matrix.shape)
    print(transpose_column_major(instance.matrix[:, 1:], instance.m).shape)
    pass
