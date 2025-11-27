import json
from pathlib import Path

from vns_with_lb import VariableNeighborhoodSearch

BENCHMARKS_FOLDER = Path(__file__).parent / "benchmarks"

ALL_INSTANCES = [
    (num_projects, num_students, instance_index)
    for num_projects in [3, 4, 5]
    for num_students in [30, 40, 50]
    for instance_index in range(10)
]


def instance_benchmark_gurobi(
    num_projects: int, num_students: int, instance_index: int, time_limit: int | float
):
    vns = VariableNeighborhoodSearch(num_projects, num_students, instance_index)
    return vns.gurobi_alone(time_limit)


def benchmark_gurobi(
    name: str,
    time_limit_per_instance: float | int,
    instances: list[tuple[int, int, int]] = ALL_INSTANCES,
):
    path = BENCHMARKS_FOLDER / "gurobi" / (name + ".json")
    if path.exists():
        raise ValueError()
    instance_solutions: dict[str, list[dict[str, int | float]]] = {}

    for instance in instances:
        key = "_".join(str(elem) for elem in instance)
        solutions = instance_benchmark_gurobi(*instance, time_limit_per_instance)
        instance_solutions[key] = solutions
        path.write_text(json.dumps(instance_solutions, indent=4), encoding="utf-8")
