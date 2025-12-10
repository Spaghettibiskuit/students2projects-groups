import random

import benchmark

ALL_INSTANCES = [
    (num_projects, num_students, instance_index)
    for num_projects in [3, 4, 5]
    for num_students in [30, 40, 50]
    for instance_index in range(10)
]

if __name__ == "__main__":
    random.seed = 0
    benchmark.benchmark(
        name="all_60s_1",
        run_gurobi=True,
        run_local_branching=True,
        run_variable_fixing=True,
        instances=ALL_INSTANCES,
    )
