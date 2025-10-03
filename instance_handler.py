"""Class which solves an instance and reports results."""

import typing

import gurobipy as gp
import pandas as pd

import base_model

type VarKey = tuple[int, int, int] | tuple[int, int] | int
type ProjectsInfo = pd.DataFrame
type StudentsInfo = pd.DataFrame
type ModelVariables = gp.tupledict[tuple[typing.Any, ...], gp.Var]


class InstanceHandler:
    """Class which solves an instance and reports results."""

    def __init__(
        self,
        instance: tuple[ProjectsInfo, StudentsInfo],
        reward_bilateral: int,
        penalty_unassigned: int,
    ):
        self.reward_bilateral = reward_bilateral
        self.penalty_unassigned = penalty_unassigned
        self.projects, self.students = instance
        self.project_ids = range(len(self.projects))
        self.student_ids = range(len(self.students))
        self.group_ids = {
            project_id: range(max_num_groups)
            for project_id, max_num_groups in enumerate(self.projects["max#groups"])
        }
        self.project_group_pairs = [
            (project_id, group_id)
            for project_id, project_group_ids in self.group_ids.items()
            for group_id in project_group_ids
        ]
        self.project_group_student_triples = [
            (project_id, group_id, student_id)
            for project_id, group_id in self.project_group_pairs
            for student_id in self.student_ids
        ]
        self.mutual_pairs = {
            (student_id, partner_id)
            for student_id, partner_ids in enumerate(self.students["fav_partners"])
            for partner_id in partner_ids
            if partner_id > student_id and student_id in self.students["fav_partners"][partner_id]
        }
        self.project_preferences = {
            (student_id, project_id): preference_value
            for student_id, preference_values in enumerate(self.students["project_prefs"])
            for project_id, preference_value in enumerate(preference_values)
        }
        self.model, self.model_variables, self.model_linear_expressions = self._build_model()

    def _build_model(
        self,
    ) -> tuple[gp.Model, dict[str, ModelVariables], dict[str, gp.LinExpr]]:
        return base_model.base_model(
            projects=self.projects,
            reward_bilateral=self.reward_bilateral,
            penalty_unassigned=self.penalty_unassigned,
            project_ids=self.project_ids,
            student_ids=self.student_ids,
            group_ids=self.group_ids,
            project_group_pairs=self.project_group_pairs,
            project_group_student_triples=self.project_group_student_triples,
            mutual_pairs=self.mutual_pairs,
            project_preferences=self.project_preferences,
        )

    def optimize(self):
        self.model.optimize()

    @property
    def objective_value(self):
        return self.model.ObjVal

    @property
    def num_unassigned(self) -> int:
        penalty_lin_exp = self.model_linear_expressions["sum_penalties_unassigned"]
        return int(penalty_lin_exp.getValue() / self.penalty_unassigned)

    @property
    def unassigned_students(self) -> list[tuple[int, str]]:
        unassigned_student_vars = self.model_variables["unassigned_students"]
        return [
            (student_id, name)
            for name, student_id, is_unassigned in zip(
                self.students["name"], self.student_ids, unassigned_student_vars.values()
            )
            if is_unassigned.X > 0.5
        ]
