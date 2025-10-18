"""Class that bundles all that is relevant in the context of a solution."""

from __future__ import annotations

from dataclasses import fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from configuration import Configuration
    from derived_modeling_data import DerivedModelingData
    from model_components import LinExpressions, Variables
    from solution_info_retriever import SolutionInformationRetriever
    from solution_viewer import SolutionViewer

SOLUTIONS_FOLDER_NAME = "solutions"


class Solution:

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        variables: Variables,
        lin_expressions: LinExpressions,
        retriever: SolutionInformationRetriever,
        viewer: SolutionViewer,
    ):
        self.config = config
        self.derived = derived
        self.variables = variables
        self.lin_expressions = lin_expressions
        self.retriever = retriever
        self.viewer = viewer

    @cached_property
    def solution_table(self):
        max_group_id = max(self.config.projects_info["max#groups"])
        students_in_groups = {}
        for group_id in range(max_group_id):
            students_in_groups[group_id] = [
                self.retriever.students_in_group(project_id, group_id)
                for project_id in self.derived.project_ids
            ]
        return pd.DataFrame(students_in_groups)

    def save_as_csv(self, filename: str, suffix: str = "csv"):
        target_folder = Path().cwd() / SOLUTIONS_FOLDER_NAME / "custom"
        path = target_folder / f"{filename}.{suffix}"
        if path.exists():
            raise ValueError("Filename already exists.")

        target_folder.mkdir(parents=True, exist_ok=True)

        top_comments = [
            f"# Objective: {self.retriever.objective_value}",
            f"# Penalty per unassigned student: {self.config.penalty_unassigned}",
            f"# Reward per materialized mutual pair: {self.config.reward_mutual_pair}",
        ]

        lin_expr_values = [
            getattr(self.lin_expressions, field.name).getValue()
            for field in fields(self.lin_expressions)
        ]
        descriptors = [
            "Sum of the realized project preferences",
            "Sum of rewards for materialized mutual pairs",
            "Sum of the penalties for leaving students unassigned",
            "Sum of penalties for surplus groups",
            "Sum of penalties for not ideal group sizes",
        ]

        middle_comments = [
            f"# {descr} is: {val:.1f}" for descr, val in zip(descriptors, lin_expr_values)
        ]

        bottom_comments = [
            f"# Materialized mutual pairs: {str(self.retriever.mutual_pairs)[1:-1]}",
            f"# Unassigned students: '{str(self.retriever.unassigned_students)[1:-1]}'",
        ]
        comments = top_comments + middle_comments + bottom_comments

        path.write_text(
            "\n".join(comments + [self.solution_table.to_csv()]),
            encoding="utf-8",
        )
