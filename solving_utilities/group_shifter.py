import functools
import itertools


class GroupShifter:

    def __init__(
        self,
        groups_only_free: set[tuple[int, int]],
        groups_mixed: set[tuple[int, int]],
        line_up_assignments: list[tuple[int, int, int]],
        project_group_student_triples: tuple[tuple[int, int, int], ...],
        assign_students_var_values: tuple[int | float, ...],
    ):
        self._groups_only_free = groups_only_free
        self._groups_mixed = groups_mixed
        self._original_line_up_assignments = line_up_assignments
        self._project_group_student_triples = project_group_student_triples
        self._assign_students_var_values = assign_students_var_values

    @functools.cached_property
    def _shifted_groups(self) -> dict[tuple[int, int], tuple[int, int]]:
        all_groups = self._groups_only_free.union(self._groups_mixed)
        if len(all_groups) != len(self._groups_only_free) + len(self._groups_mixed):
            raise ValueError()

        all_groups_ordered = sorted(all_groups)
        groups: dict[int, list[int]] = {}
        for project_id, group_id in all_groups_ordered:
            groups.setdefault(project_id, []).append(group_id)

        groups_only_free_ordered = sorted(self._groups_only_free)
        affected_projects = [project_id for project_id, _ in groups_only_free_ordered]
        affected_groups = {project_id: groups[project_id] for project_id in affected_projects}

        only_free_affected_groups: dict[int, list[int]] = {}
        for project_id, group_id in groups_only_free_ordered:
            only_free_affected_groups.setdefault(project_id, []).append(group_id)

        mixed_affected_groups = {
            project_id: [
                group_id
                for group_id in group_ids
                if (project_id, group_id) not in self._groups_only_free
            ]
            for project_id, group_ids in affected_groups.items()
        }

        return {
            (project_id, group_id): (project_id, new_group_id)
            for (project_id, mixed_affected_groups), only_free_affected_groups in zip(
                mixed_affected_groups.items(), only_free_affected_groups.values()
            )
            for group_id, new_group_id in zip(
                mixed_affected_groups + only_free_affected_groups,
                itertools.count(),
            )
        }

    @property
    def adjusted_line_up_assignments(self) -> list[tuple[int, int, int]]:
        shifted_groups = self._shifted_groups
        return [
            (
                (*shifted_group, student_id)
                if (shifted_group := shifted_groups.get((proj_id, group_id))) is not None
                else (proj_id, group_id, student_id)
            )
            for proj_id, group_id, student_id in self._original_line_up_assignments
        ]

    @property
    def adjusted_start_values(self) -> list[int | float]:
        shifted_groups = self._shifted_groups
        start_values = dict(
            zip(
                self._project_group_student_triples,
                self._assign_students_var_values,
            )
        )
        for project_id, group_id, student_id in self._original_line_up_assignments:
            if (new_group := shifted_groups.get((project_id, group_id))) is not None:
                old = (project_id, group_id, student_id)
                new = (*new_group, student_id)
                start_values[old], start_values[new] = start_values[new], start_values[old]

        return list(start_values.values())
