class BranchingConstraintsEfficacyChecker:

    def __init__(self):
        self.optimal_within_branching_constraints: dict[tuple[int, ...], bool] = {}

    def check(self, solution_variables: tuple[int | float, ...], is_optimal_within_bounds: bool):
        rounded = tuple(round(var) for var in solution_variables)
        if rounded in self.optimal_within_branching_constraints:
            was_optimal_within_bounds = self.optimal_within_branching_constraints[rounded]

            # too strict in general case, works in particular case
            if was_optimal_within_bounds is False and is_optimal_within_bounds is False:
                raise ValueError(
                    "Solution previously found as not optimal found again as not optimal!"
                )

            if was_optimal_within_bounds is False and is_optimal_within_bounds is True:
                self.optimal_within_branching_constraints[rounded] = True

            if was_optimal_within_bounds is True and is_optimal_within_bounds is False:
                raise ValueError(
                    "Solution previously found as optimal found again as not optimal!"
                )

            if was_optimal_within_bounds is True and is_optimal_within_bounds is True:
                raise ValueError("Solution previously found as optimal found again as optimal!")

        else:
            self.optimal_within_branching_constraints[rounded] = is_optimal_within_bounds
