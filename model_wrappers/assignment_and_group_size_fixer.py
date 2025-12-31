import functools

from gurobipy import GRB

from model_wrappers.assignment_fixer import AssignmentFixer


class AssignmentAndGroupSizeFixer(AssignmentFixer):

    @functools.cached_property
    def _len_project_group_pairs_zeros(self) -> list[float]:
        return [0.0] * len(self.derived.project_group_pairs)

    @functools.cached_property
    def _len_project_group_pairs_undefineds(self) -> list[float]:
        return [GRB.UNDEFINED] * len(self.derived.project_group_pairs)

    def fix_group_sizes(self):
        for grb_vars, values in [
            (
                self.variable_access.group_size_surplus,
                self.current_solution.group_size_surplus_var_values,
            ),
            (
                self.variable_access.group_size_deficit,
                self.current_solution.group_size_deficit_var_values,
            ),
        ]:
            for attribute, values in ["LB", "UB"]:
                self.model.setAttr(attribute, grb_vars, values)

    def release_group_size(self):
        for grb_vars in [
            self.variable_access.group_size_surplus,
            self.variable_access.group_size_deficit,
        ]:
            for attribute, boundary_vals in [
                ("LB", self._len_project_group_pairs_zeros),
                ("UB", self._len_project_group_pairs_undefineds),
            ]:
                self.model.setAttr(attribute, grb_vars, boundary_vals)

    def fix_rest(self, zone_a: int, zone_b: int, num_zones: int):
        line_up_assignments = self.current_sol_fixing_data.line_up_assignments
        free_assignments, _ = self._separate_assignments(
            zone_a, zone_b, num_zones, line_up_assignments
        )
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
