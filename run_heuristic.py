import os
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

# Ignore warnings from numba
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from utils import import_instance
from sclocalsearch import scLocalSearch


if __name__ == "__main__":

    paths = [
        os.path.join("instances", "rail507"),
        os.path.join("instances", "rail516"),
        os.path.join("instances", "rail582"),
        os.path.join("instances", "rail2536"),
        os.path.join("instances", "rail2586"),
        os.path.join("instances", "rail4284"),
        os.path.join("instances", "rail4872"),
    ]

    for i, instance_path in enumerate(paths):

        instance = import_instance(instance_path)

        # delete results from previous runs
        sol_path = os.path.join("solutions", instance.name)
        [
            os.remove(os.path.join(sol_path, file))
            for file in os.listdir(sol_path)
            if file.endswith(".sol")
        ]

        start_time = time.time()
        solution, cost = scLocalSearch(
            instance,
            start_time=start_time,
            alpha=50,
            max_patience=2000,
            time_limit=600,
            seed=0,
        )
