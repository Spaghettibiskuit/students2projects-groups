"""Class which solves an instance and reports results."""

import itertools as it
import statistics
import typing

import gurobipy as gp
import pandas as pd

import base_model

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
        return round(penalty_lin_exp.getValue() / self.penalty_unassigned)

    @property
    def unassigned_students(self) -> list[tuple[int, str]]:
        unassigned_student_vars = self.model_variables["unassigned_students"]
        return [
            (student_id, name)
            for name, student_id, is_unassigned in zip(
                self.students["name"], self.student_ids, unassigned_student_vars.values()
            )
            if round(is_unassigned.X)
        ]

    def num_students_in_group(self, project_id: int, group_id: int) -> int:
        assignment_vars = self.model_variables["assign_students"]
        return round(assignment_vars.sum(project_id, group_id, "*").getValue())

    def students_in_group(self, project_id: int, group_id: int) -> list[int]:
        assignment_vars = self.model_variables["assign_students"]
        return [
            student_id
            for student_id in self.student_ids
            if round(assignment_vars[project_id, group_id, student_id].X)
        ]

    def students_in_group_names(self, project_id: int, group_id: int) -> list[str]:
        return [
            self.students["name"][student_id]
            for student_id in self.students_in_group(project_id, group_id)
        ]

    def pref_vals_students_in_group(self, project_id: int, group_id: int) -> dict[int, int]:
        return {
            student_id: self.project_preferences[student_id, project_id]
            for student_id in self.students_in_group(project_id, group_id)
        }

    def mutual_pairs_in_group(self, project_id: int, group_id: int) -> set[tuple[int, int]]:
        return self.mutual_pairs.intersection(
            pair for pair in it.combinations(self.students_in_group(project_id, group_id), 2)
        )

    def num_mutual_pairs_in_group(self, project_id: int, group_id: int) -> int:
        return sum(
            pair in self.mutual_pairs
            for pair in it.combinations(self.students_in_group(project_id, group_id), 2)
        )

    def mutual_pairs_in_group_names(self, project_id: int, group_id: int) -> set[tuple[str, str]]:
        return {
            (self.students["name"][first_id], self.students["name"][second_id])
            for first_id, second_id in self.mutual_pairs_in_group(project_id, group_id)
        }

    @property
    def solution_summary(self) -> pd.DataFrame:
        all_summary_tables = [
            self.summary_table_project(project_id) for project_id in self.project_ids
        ]
        open_projects = {
            project_id: summary_table
            for project_id, summary_table in zip(self.project_ids, all_summary_tables)
            if not summary_table.empty
        }
        summary_tables = open_projects.values()

        group_quantities = [len(summary_table) for summary_table in summary_tables]
        student_quantities = [sum(summary_table["#students"]) for summary_table in summary_tables]
        data = {}
        data["ID"] = open_projects.keys()
        data["#groups"] = group_quantities
        data["max_size"] = [max(summary_table["#students"]) for summary_table in summary_tables]
        data["min_size"] = [min(summary_table["#students"]) for summary_table in summary_tables]
        data["mean_size"] = [
            num_students / num_groups
            for num_students, num_groups in zip(student_quantities, group_quantities)
        ]
        data["max_pref"] = [max(summary_table["max_pref"]) for summary_table in summary_tables]
        data["min_pref"] = [min(summary_table["min_pref"]) for summary_table in summary_tables]
        data["mean_pref"] = [
            (summary_table["#students"] * summary_table["mean_pref"]).sum() / num_students
            for num_students, summary_table in zip(student_quantities, summary_tables)
        ]
        data["#mutual_pairs"] = [
            sum(summary_table["#mutual_pairs"]) for summary_table in summary_tables
        ]

        return pd.DataFrame(data)

    def summary_table_project(self, project_id: int) -> pd.DataFrame:
        student_quantities = [
            num_students
            for group_id in self.group_ids[project_id]
            if (num_students := self.num_students_in_group(project_id, group_id))
        ]
        group_ids = list(range(len(student_quantities)))
        pref_vals_in_groups = [
            list(self.pref_vals_students_in_group(project_id, group_id).values())
            for group_id in group_ids
        ]
        data = {}
        data["#students"] = student_quantities
        data["max_pref"] = [max(pref_vals) for pref_vals in pref_vals_in_groups]
        data["min_pref"] = [min(pref_vals) for pref_vals in pref_vals_in_groups]
        data["mean_pref"] = [statistics.mean(pref_vals) for pref_vals in pref_vals_in_groups]
        data["#mutual_pairs"] = [
            self.num_mutual_pairs_in_group(project_id, group_id) for group_id in group_ids
        ]
        return pd.DataFrame(data)
