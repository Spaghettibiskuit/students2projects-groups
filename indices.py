"""A data class that contains all indices relevant for modeling the problem instance."""

from dataclasses import dataclass

from configurator import Configuration


@dataclass(frozen=True)
class Indices:
    project_ids: range
    student_ids: range
    group_ids: dict[int, range]
    project_group_pairs: tuple[tuple[int, int], ...]
    project_group_student_triples: tuple[tuple[int, int, int], ...]


def get_indices(config: Configuration) -> Indices:
    project_ids = range(len(config.projects_info))
    student_ids = range(len(config.students_info))
    group_ids = {
        project_id: range(max_num_groups)
        for project_id, max_num_groups in enumerate(config.projects_info["max#groups"])
    }
    project_group_pairs = tuple(
        (project_id, group_id)
        for project_id, project_group_ids in group_ids.items()
        for group_id in project_group_ids
    )
    project_group_student_triples = tuple(
        (project_id, group_id, student_id)
        for project_id, group_id in project_group_pairs
        for student_id in student_ids
    )

    return Indices(
        project_ids, student_ids, group_ids, project_group_pairs, project_group_student_triples
    )
