import dataclasses
import json
from pathlib import Path

import pandas
from gurobipy import GRB


@dataclasses.dataclass
class InstanceSummaryHeuristic:
    best_objective: int
    runtime: float


@dataclasses.dataclass
class InstanceSummaryGurobi:
    best_objective: int
    best_bound: int
    runtime: float


def instance_summary_heuristic(
    results: dict[str, list[dict[str, int | float | str]]],
    num_projects: int,
    num_students: int,
    instance_index: int,
) -> InstanceSummaryHeuristic:
    summaries = results[f"{num_projects}_{num_students}_{instance_index}"]
    correctness_indicator = summaries.pop()
    if not bool(correctness_indicator["is_correct"]):
        raise ValueError()

    best_objective = -GRB.MAXINT
    runtime_best_objective = 0
    for summary in summaries:

        curr_objective = summary["objective"]
        curr_runtime = summary["runtime"]

        if not isinstance(curr_objective, int) or not isinstance(curr_runtime, float):
            raise TypeError()
        if curr_objective < best_objective or curr_runtime < runtime_best_objective:
            raise ValueError()

        if curr_objective > best_objective:
            best_objective = curr_objective
            runtime_best_objective = curr_runtime

    return InstanceSummaryHeuristic(best_objective, runtime_best_objective)


def instance_summary_gurobi(
    results: dict[str, list[dict[str, int | float | str]]],
    num_projects: int,
    num_students: int,
    instance_index: int,
) -> InstanceSummaryGurobi:
    summaries = results[f"{num_projects}_{num_students}_{instance_index}"]
    correctness_indicator = summaries.pop()
    if not bool(correctness_indicator["is_correct"]):
        raise ValueError()

    best_objective = -GRB.MAXINT
    best_bound = GRB.MAXINT
    runtime_best_objective = 0

    for summary in summaries:

        curr_objective = summary["objective"]
        curr_bound = summary["bound"]
        curr_runtime = summary["runtime"]

        if (
            not isinstance(curr_objective, int)
            or not isinstance(curr_bound, int)
            or not isinstance(curr_runtime, float)
        ):
            raise TypeError()
        if (
            curr_objective < best_objective
            or curr_bound > best_bound
            or curr_runtime < runtime_best_objective
        ):
            raise ValueError()

        if curr_objective > best_objective:
            best_objective = curr_objective
            runtime_best_objective = curr_runtime
        best_bound = min(best_bound, curr_bound)

    return InstanceSummaryGurobi(best_objective, best_bound, runtime_best_objective)


def instance_summaries_heuristic(
    results: dict[str, list[dict[str, int | float | str]]],
    num_projects: int,
    num_students: int,
    instance_indexes: range,
) -> list[InstanceSummaryHeuristic]:
    return [
        instance_summary_heuristic(results, num_projects, num_students, instance_index)
        for instance_index in instance_indexes
    ]


def instance_summaries_gurobi(
    results: dict[str, list[dict[str, int | float | str]]],
    num_projects: int,
    num_students: int,
    instance_indexes: range,
) -> list[InstanceSummaryGurobi]:
    return [
        instance_summary_gurobi(results, num_projects, num_students, instance_index)
        for instance_index in instance_indexes
    ]


def granular_all_methods(
    num_projects: int,
    num_students: int,
    instance_indexes: range,
    gurobi_path: Path,
    lb_path: Path,
    vf_path: Path,
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    gurobi_res = json.loads(gurobi_path.read_text("utf-8"))
    lb_res = json.loads(lb_path.read_text("utf-8"))
    vf_res = json.loads(vf_path.read_text("utf-8"))

    gurobi_summaries = instance_summaries_gurobi(
        gurobi_res, num_projects, num_students, instance_indexes
    )
    lb_summaries = instance_summaries_heuristic(
        lb_res, num_projects, num_students, instance_indexes
    )
    vf_summaries = instance_summaries_heuristic(
        vf_res, num_projects, num_students, instance_indexes
    )
    bests_grb = [summary.best_objective for summary in gurobi_summaries]
    bests_lb = [summary.best_objective for summary in lb_summaries]
    bests_vf = [summary.best_objective for summary in vf_summaries]
    bounds = [summary.best_bound for summary in gurobi_summaries]

    abs_results = {
        "Gurobi": bests_grb,
        "Local Branching": bests_lb,
        "Variable Fixing": bests_vf,
        "Best bound": bounds,
    }
    runtime_results = {
        "Gurobi": [summary.runtime for summary in gurobi_summaries],
        "Local Branching": [summary.runtime for summary in lb_summaries],
        "Variable Fixing": [summary.runtime for summary in vf_summaries],
    }
    gap_results: dict[str, list[float]] = {
        "Gurobi": [],
        "Local Branching": [],
        "Variable Fixing": [],
    }
    for best_grb, best_lb, best_vf, bound in zip(bests_grb, bests_lb, bests_vf, bounds):
        gap_results["Gurobi"].append((bound - best_grb) / bound)
        gap_results["Local Branching"].append((bound - best_lb) / bound)
        gap_results["Variable Fixing"].append((bound - best_vf) / bound)

    return (
        pandas.DataFrame(abs_results),
        pandas.DataFrame(runtime_results),
        pandas.DataFrame(gap_results),
    )
