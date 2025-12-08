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
        name="5_50_4_180s",
        run_gurobi=False,
        run_local_branching=False,
        run_variable_fixing=True,
        instances=ALL_INSTANCES[84:85],
        variable_fixing_paramters=benchmark.VariableFixingParamters(total_time_limit=180),
    )
