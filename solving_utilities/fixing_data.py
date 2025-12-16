import random
from dataclasses import dataclass

import gurobipy

from modeling.configuration import Configuration
from modeling.derived_modeling_data import DerivedModelingData
from modeling.model_components import LinExpressions, Variables
from solution_processing.individual_assignment_scorer import IndividualAssignmentScorer


@dataclass(frozen=True)
class FixingByRankingData:

    scores: dict[tuple[int, int, int], float]
    ranked_assignments: list[tuple[int, int, int]]
    assignments: set[tuple[int, int, int]]
    line_up_assignments: list[tuple[int, int, int]]
    line_up_ids: list[int]

    @classmethod
    def get(
        cls,
        config: Configuration,
        derived: DerivedModelingData,
        variables: Variables,
        lin_expressions: LinExpressions,
        model: gurobipy.Model,
    ):
        scores = IndividualAssignmentScorer(
            config, derived, variables, lin_expressions
        ).assignment_scores
        ranked_assignments = sorted(scores.keys(), key=lambda k: scores[k])
        assignments = set(ranked_assignments)

        num_unassigned = config.number_of_students - len(ranked_assignments)

        derived_obj_val = sum(scores.values()) - num_unassigned * config.penalty_unassigned
        if abs(derived_obj_val - model.ObjVal) > 1e-1:
            raise ValueError(f"derived {derived_obj_val} is not real {model.ObjVal}")
        if num_unassigned > 0:
            assigned_ids = set(student_id for _, _, student_id in ranked_assignments)
            unassigned_ids = assigned_ids.difference(derived.student_ids)

            line_up_assignments = fixing_line_up_assignments(
                config, derived, ranked_assignments, unassigned_ids
            )
        else:
            line_up_assignments = ranked_assignments

        line_up_ids = [student_id for _, _, student_id in line_up_assignments]

        return cls(
            scores=scores,
            ranked_assignments=ranked_assignments,
            assignments=assignments,
            line_up_assignments=line_up_assignments,
            line_up_ids=line_up_ids,
        )


def fixing_line_up_assignments(
    config: Configuration,
    derived: DerivedModelingData,
    ranked_assignments: list[tuple[int, int, int]],
    unassigned_ids: set[int],
):

    pseudo_assignments = ((-1, -1, student_id) for student_id in unassigned_ids)
    actual_assignments = iter(ranked_assignments)

    positions = set(random.sample(derived.student_ids, k=len(unassigned_ids)))

    line_up: list[tuple[int, int, int]] = []

    for i in range(config.number_of_students):
        if i in positions:
            line_up.append(next(pseudo_assignments))
        else:
            line_up.append(next(actual_assignments))

    if len(line_up) != config.number_of_students:
        raise ValueError("Length does not match.")

    return line_up
