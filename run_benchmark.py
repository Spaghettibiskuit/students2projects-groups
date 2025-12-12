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
        name="5_even_bigger_1h",
        run_gurobi=True,
        run_local_branching=False,
        run_variable_fixing=False,
        instances=[(60, 600, 0), (70, 700, 0), (80, 800, 0), (90, 900, 0), (100, 1_000, 0)],
        gurobi_alone_parameters=benchmark.GurobiAloneParameters(time_limit=3_600),
        # local_branching_parameters=benchmark.LocalBranchingParameters(total_time_limit=120),
        # variable_fixing_paramters=benchmark.VariableFixingParamters(total_time_limit=120),
    )
