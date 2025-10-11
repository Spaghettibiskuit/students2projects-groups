"""Functions for loading an instance."""

import json
from pathlib import Path

import pandas as pd

type ProjectsInfo = pd.DataFrame
type StudentsInfo = pd.DataFrame


def load_instance(
    num_projects: int, num_students: int, instance_index: int
) -> tuple[ProjectsInfo, StudentsInfo]:
    """Return the specified instance of the SPAwGBP."""
    folder = Path("instances")
    subfolder = f"{num_projects}_projects_{num_students}_students"
    shared_prefix = f"generic_{num_projects}_{num_students}"
    shared_suffix = f"{instance_index}.csv"
    path_projects = folder / subfolder / f"{shared_prefix}_projects_{shared_suffix}"
    path_students = folder / subfolder / f"{shared_prefix}_students_{shared_suffix}"
    projects: pd.DataFrame = pd.read_csv(path_projects)  # type: ignore
    students: pd.DataFrame = pd.read_csv(path_students)  # type: ignore
    students["fav_partners"] = students["fav_partners"].apply(lambda x: frozenset(json.loads(x)))  # type: ignore
    students["project_prefs"] = students["project_prefs"].apply(lambda x: tuple(json.loads(x)))  # type: ignore
    return projects, students
