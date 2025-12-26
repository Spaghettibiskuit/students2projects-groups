import dataclasses
import enum
import json
import random
from pathlib import Path

import utilities
from variable_neighborhood_search import VariableNeighborhoodSearch

BENCHMARKS_FOLDER = Path(__file__).parent / "benchmarks"


class Subfolders(enum.StrEnum):
    GUROBI = "gurobi"
    LOCAL_BRANCHING = "local_branching"
    VARIABLE_FIXING = "variable_fixing"


@dataclasses.dataclass
class LocalBranchingParameters:
    total_time_limit: int | float = 60
    k_min_perc: int | float = 10
    k_step_perc: int | float = 10
    k_max_perc: int | float = 80
    l_min_perc: int | float = 10
    l_step_perc: int | float = 10
    l_max_perc: int | float = 40
    initial_patience: float | int = 3
    shake_patience: float | int = 2
    min_optimization_patience: int | float = 3
    step_optimization_patience: int | float = 1
    drop_branching_constrs_before_shake: bool = False


@dataclasses.dataclass
class VariableFixingParamters:
    total_time_limit: int | float = 60
    min_num_zones: int = 4
    step_num_zones: int = 1
    max_num_zones: int = 6
    max_iterations_per_num_zones: int = 20
    min_shake_perc: int = 10
    step_shake_perc: int = 10
    max_shake_perc: int = 80
    initial_patience: int | float = 3
    shake_patience: int | float = 2
    min_optimization_patience: int | float = 1
    step_optimization_patience: int | float = 1


@dataclasses.dataclass
class GurobiAloneParameters:
    time_limit: int | float = 60


def check_whether_instances_exist(instances: list[tuple[int, int, int]]):
    for instance in instances:
        path_projects, path_students = utilities.build_paths(*instance)
        if not path_projects.exists():
            raise ValueError(f"{repr(path_projects)} does not exist")
        if not path_students.exists():
            raise ValueError(f"{repr(path_students)} does not exist")


def get_path(subfolder: str, name: str):
    return BENCHMARKS_FOLDER / subfolder / (name + ".json")


def benchmark_instance_gurobi_alone(
    instance: tuple[int, int, int], parameters: GurobiAloneParameters
) -> list[dict[str, int | float]]:
    vns = VariableNeighborhoodSearch(*instance)
    return vns.gurobi_alone(**dataclasses.asdict(parameters))


def benchmark_instance_local_branching(
    instance: tuple[int, int, int], parameters: LocalBranchingParameters
) -> list[dict[str, int | float | str]]:
    vns = VariableNeighborhoodSearch(*instance)
    return vns.run_vns_with_lb(**dataclasses.asdict(parameters))


def benchmark_instance_variable_fixing(
    instance: tuple[int, int, int], parameters: VariableFixingParamters
) -> list[dict[str, int | float | str]]:
    vns = VariableNeighborhoodSearch(*instance)
    return vns.run_vns_with_var_fixing(**dataclasses.asdict(parameters))


def benchmark(
    name: str,
    run_gurobi: bool,
    run_local_branching: bool,
    run_variable_fixing: bool,
    instances: list[tuple[int, int, int]],
    gurobi_alone_parameters: GurobiAloneParameters = GurobiAloneParameters(),
    local_branching_parameters: LocalBranchingParameters = LocalBranchingParameters(),
    variable_fixing_paramters: VariableFixingParamters = VariableFixingParamters(),
    seed: int = 0,
):
    check_whether_instances_exist(instances)

    gurobi_path = get_path(Subfolders.GUROBI, name)
    local_branching_path = get_path(Subfolders.LOCAL_BRANCHING, name)
    variable_fixing_path = get_path(Subfolders.VARIABLE_FIXING, name)

    for path, will_be_written_to in (
        (gurobi_path, run_gurobi),
        (local_branching_path, run_local_branching),
        (variable_fixing_path, run_variable_fixing),
    ):
        if path.exists() and will_be_written_to:
            raise ValueError(f"{repr(path)} that will be written to already exists.")

    gurobi_solutions: dict[str, list[dict[str, int | float]]] = {}
    local_branching_solutions: dict[str, list[dict[str, int | float | str]]] = {}
    variable_fixing_solutions: dict[str, list[dict[str, int | float | str]]] = {}

    for instance in instances:

        key = "_".join(str(elem) for elem in instance)

        if run_gurobi:
            random.seed(seed)
            gurobi_solutions[key] = benchmark_instance_gurobi_alone(
                instance, gurobi_alone_parameters
            )
            gurobi_path.write_text(json.dumps(gurobi_solutions, indent=4), encoding="utf-8")

        if run_local_branching:
            random.seed(seed)
            local_branching_solutions[key] = benchmark_instance_local_branching(
                instance, local_branching_parameters
            )
            local_branching_path.write_text(
                json.dumps(local_branching_solutions, indent=4), encoding="utf-8"
            )

        if run_variable_fixing:
            random.seed(seed)
            variable_fixing_solutions[key] = benchmark_instance_variable_fixing(
                instance, variable_fixing_paramters
            )
            variable_fixing_path.write_text(
                json.dumps(variable_fixing_solutions, indent=4), encoding="utf-8"
            )


if __name__ == "__main__":
    # all_small_instances = [
    #     (num_projects, num_students, instance_index)
    #     for num_projects in [3, 4, 5]
    #     for num_students in [30, 40, 50]
    #     for instance_index in range(10)
    # ]
    benchmark(
        name="20_60s_2",
        run_gurobi=False,
        run_local_branching=False,
        run_variable_fixing=True,
        instances=[(i * 10, i * 100, j) for i in range(2, 3) for j in range(10)],
        # gurobi_alone_parameters=GurobiAloneParameters(time_limit=300),
        # local_branching_parameters=LocalBranchingParameters(total_time_limit=300),
        variable_fixing_paramters=VariableFixingParamters(total_time_limit=60),
    )
