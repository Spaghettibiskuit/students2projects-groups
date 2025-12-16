"""A class that allows to view the soltution of a Gurobi model."""

import statistics
from functools import cached_property, lru_cache

import pandas as pd

from modeling.derived_modeling_data import DerivedModelingData
from solution_processing.solution_info_retriever import SolutionInformationRetriever


class SolutionViewer:

    def __init__(self, derived: DerivedModelingData, retriever: SolutionInformationRetriever):
        self.derived = derived
        self.retriever = retriever

    @cached_property
    def solution_summary(self) -> pd.DataFrame:
        open_projects = {
            project_id: summary_table
            for project_id in self.derived.project_ids
            if not (summary_table := self.summary_table_project(project_id)).empty
        }
        summary_tables = open_projects.values()

        group_quantities = [len(summary_table) for summary_table in summary_tables]
        student_quantities = [sum(summary_table["#students"]) for summary_table in summary_tables]

        result = {}
        result["ID"] = open_projects.keys()
        result["#groups"] = group_quantities
        result["max_size"] = [max(summary_table["#students"]) for summary_table in summary_tables]
        result["min_size"] = [min(summary_table["#students"]) for summary_table in summary_tables]
        result["mean_size"] = [
            num_students / num_groups
            for num_students, num_groups in zip(student_quantities, group_quantities)
        ]
        result["max_pref"] = [max(summary_table["max_pref"]) for summary_table in summary_tables]
        result["min_pref"] = [min(summary_table["min_pref"]) for summary_table in summary_tables]
        result["mean_pref"] = [
            (summary_table["#students"] * summary_table["mean_pref"]).sum() / num_students
            for num_students, summary_table in zip(student_quantities, summary_tables)
        ]
        result["#mutual_pairs"] = [
            sum(summary_table["#mutual_pairs"]) for summary_table in summary_tables
        ]

        return pd.DataFrame(result)

    @lru_cache(maxsize=128)
    def summary_table_project(self, project_id: int) -> pd.DataFrame:
        student_quantities = [
            num_students
            for group_id in self.derived.group_ids[project_id]
            if (num_students := len(self.retriever.students_in_group(project_id, group_id)))
        ]
        group_ids = list(range(len(student_quantities)))
        pref_vals_in_groups = [
            list(self.retriever.pref_vals_students_in_group(project_id, group_id).values())
            for group_id in group_ids
        ]
        result = {}
        result["#students"] = student_quantities
        result["max_pref"] = [max(pref_vals) for pref_vals in pref_vals_in_groups]
        result["min_pref"] = [min(pref_vals) for pref_vals in pref_vals_in_groups]
        result["mean_pref"] = [statistics.mean(pref_vals) for pref_vals in pref_vals_in_groups]
        result["#mutual_pairs"] = [
            len(self.retriever.mutual_pairs_in_group(project_id, group_id))
            for group_id in group_ids
        ]
        return pd.DataFrame(result)
