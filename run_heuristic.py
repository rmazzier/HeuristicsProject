import os
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from utils import import_instance, Instance, save_solution_to_file
from sclocalsearch import scLocalSearch


if __name__ == "__main__":

    paths = [
        os.path.join("rail", "instances", "rail507"),
        # os.path.join("rail", "instances", "rail516"),
        # os.path.join("rail", "instances", "rail582"),
        # os.path.join("rail", "instances", "rail2536"),
        # os.path.join("rail", "instances", "rail2586"),
        # os.path.join("rail", "instances", "rail4284"),
        # os.path.join("rail", "instances", "rail4872"),
    ]

    patience_values = [
        2000,
        2000,
        2000,
        400,
        400,
        400,
        400,
    ]
    for i, instance_path in enumerate(paths):

        start_time = time.time()
        instance = import_instance(instance_path)
        solution, cost = scLocalSearch(
            instance,
            start_time=start_time,
            alpha=50,
            max_patience=patience_values[i],
            time_limit=600,
            seed=0,
        )

        # save_solution_to_file(
        #     solution,
        #     instance,
        #     f"./solutions/localsearch/{instance.name}.0.sol",
        # )
