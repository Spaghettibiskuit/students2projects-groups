import json
import random
from pathlib import Path

from vns_with_lb import VariableNeighborhoodSearch

random.seed = 0


BENCHMARKS_FOLDER = Path(__file__).parent / "benchmarks"

ALL_INSTANCES = [
    (num_projects, num_students, instance_index)
    for num_projects in [3, 4, 5]
    for num_students in [30, 40, 50]
    for instance_index in range(10)
]

GUROBI = 0
LOCAL_BRANCHING = 1
VARIABLE_FIXING = 2

SUBFOLDERS = {
    GUROBI: "gurobi",
    LOCAL_BRANCHING: "local_branching",
    VARIABLE_FIXING: "variable_fixing",
}


def instance_benchmark(
    num_projects: int, num_students: int, instance_index: int, time_limit: int | float, method: int
):
    vns = VariableNeighborhoodSearch(num_projects, num_students, instance_index)
    if method == GUROBI:
        return vns.gurobi_alone(time_limit)
    if method == LOCAL_BRANCHING:
        return vns.run_vns_with_lb(total_time_limit=time_limit)
    if method == VARIABLE_FIXING:
        return vns.run_vns_with_var_fixing(total_time_limit=time_limit)
    raise ValueError()


def benchmark_method(
    name: str,
    method: int,
    time_limit_per_instance: float | int,
    instances: list[tuple[int, int, int]] = ALL_INSTANCES,
):
    if (subfolder := SUBFOLDERS.get(method)) is None:
        raise ValueError()

    path = BENCHMARKS_FOLDER / subfolder / (name + ".json")
    if path.exists():
        raise ValueError()
    instance_solutions = {}  # type: ignore

    for instance in instances:
        key = "_".join(str(elem) for elem in instance)
        solutions = instance_benchmark(*instance, time_limit_per_instance, method)
        instance_solutions[key] = solutions
        path.write_text(json.dumps(instance_solutions, indent=4), encoding="utf-8")


def benchmark_all(
    name: str,
    time_limit_per_instance: int | float,
    instances: list[tuple[int, int, int]] = ALL_INSTANCES,
):
    for method in SUBFOLDERS:
        benchmark_method(name, method, time_limit_per_instance, instances)
