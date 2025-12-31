"""A class that contains a model which it reduces according to VNS rules."""

from model_wrappers.assignment_fixer import AssignmentFixer
from model_wrappers.thin_wrappers import AssignmentFixerInitializer
from solving_utilities.group_shifter import GroupShifter


class OnlyAssignmentFixer(AssignmentFixer):
    """Contains a model further constrained for local branching."""

    def _separate_groups(
        self,
        free_assignments: list[tuple[int, int, int]],
        fixed_assignments: list[tuple[int, int, int]],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        groups_of_free = set(
            (project_id, group_id)
            for project_id, group_id, _ in free_assignments
            if project_id != -1  # Must be real group
        )
        groups_of_fixed = set(
            (project_id, group_id)
            for project_id, group_id, _ in fixed_assignments
            if project_id != -1
        )
        return groups_of_free.difference(groups_of_fixed), groups_of_fixed

    def fix_rest(self, zone_a: int, zone_b: int, num_zones: int):
        line_up_assignments = self.current_sol_fixing_data.line_up_assignments
        free_assignments, fixed_assignments = self._separate_assignments(
            zone_a, zone_b, num_zones, line_up_assignments
        )
        groups_only_free, groups_mixed = self._separate_groups(free_assignments, fixed_assignments)
        if groups_only_free:
            group_shifter = GroupShifter(
                groups_only_free=groups_only_free,
                groups_mixed=groups_mixed,
                line_up_assignments=self.current_sol_fixing_data.line_up_assignments,
                project_group_student_triples=self.derived.project_group_student_triples,
                assign_students_var_values=self.current_solution.assign_students_var_values,
            )
            line_up_assignments = group_shifter.adjusted_line_up_assignments
            start_values = group_shifter.adjusted_start_values
            free_assignments, fixed_assignments = self._separate_assignments(
                zone_a, zone_b, num_zones, line_up_assignments
            )
            assignments = set(free_assignments + fixed_assignments)
        else:
            start_values = self.current_solution.assign_students_var_values
            assignments = self.current_sol_fixing_data.assignments

        self.model.setAttr("Start", self.variable_access.assign_students, start_values)

        free_student_ids = set(student_id for _, _, student_id in free_assignments)
        upper_bounds = [
            (
                1
                if student_id in free_student_ids
                or (project_id, group_id, student_id) in assignments
                else 0
            )
            for project_id, group_id, student_id in self.derived.project_group_student_triples
        ]
        self.model.setAttr("UB", self.variable_access.assign_students, upper_bounds)

    @classmethod
    def get(cls, initializer: AssignmentFixerInitializer):
        return cls(
            config=initializer.config,
            derived=initializer.derived,
            model_components=initializer.model_components,
            model=initializer.model,
            sol_reminder=initializer.current_solution,
            fixing_data=initializer.fixing_data,
            start_time=initializer.start_time,
            solution_summaries=initializer.solution_summaries,
        )
