from time import time

from gurobipy import GRB

from branching_constraints_efficacy_checker import BranchingConstraintsEfficacyChecker
from configuration import Configuration
from constrained_model import ConstrainedModel
from derived_modeling_data import DerivedModelingData
from solution import Solution
from solution_checker import SolutionChecker
from solution_info_retriever import SolutionInformationRetriever
from solution_viewer import SolutionViewer


class VariableNeighborhoodSearch:

    def __init__(
        self,
        number_of_projects: int,
        number_of_students: int,
        instance_index: int,
        reward_mutual_pair: int,
        penalty_unassigned: int,
    ):
        self.config = Configuration.get(
            number_of_projects=number_of_projects,
            number_of_students=number_of_students,
            instance_index=instance_index,
            reward_mutual_pair=reward_mutual_pair,
            penalty_unassigned=penalty_unassigned,
        )
        self.derived = DerivedModelingData.get(config=self.config)
        self.best_model = None
        self.best_solution = None

    def gurobi_alone(self, time_limit: int | float = float("inf")) -> ConstrainedModel:
        active_model = ConstrainedModel(
            config=self.config,
            derived=self.derived,
        )
        active_model.model.Params.TimeLimit = time_limit
        active_model.model.optimize()

        self.best_model = active_model
        self._post_processing()
        return self.best_model

    def run_vns_with_lb_basic(
        self, total_time_limit: int | float, node_time_limit: int | float, k_step: int
    ):
        model = ConstrainedModel(
            config=self.config,
            derived=self.derived,
        )
        start_time = time()
        k_cur = k_step
        model.set_solution_limit(1)
        time_limit = total_time_limit - (time() - start_time)
        model.set_time_limit(max(0, time_limit))
        model.optimize()
        model.store_solution()
        model.store_last_feasible_solution_as_incumbent()

        while time() - start_time < total_time_limit:
            rhs = 1
            model.eliminate_solution_limit()
            branching_constraints_efficacy_checker = BranchingConstraintsEfficacyChecker()
            while time() - start_time < total_time_limit:
                model.set_branching_var_values_for_vnd()
                model.add_bounding_branching_constraint(rhs)
                time_limit = min(node_time_limit, total_time_limit - (time() - start_time))
                model.set_time_limit(max(0, time_limit))
                model.set_cutoff(ascending=False)
                model.optimize()
                status_code = model.status
                if status_code in (GRB.INFEASIBLE, GRB.CUTOFF):
                    model.pop_branching_constraints_stack()
                    model.add_excluding_branching_constraint(rhs)
                    rhs += 1

                elif status_code == GRB.OPTIMAL:
                    model.store_solution()
                    model.pop_branching_constraints_stack()
                    model.add_excluding_branching_constraint(rhs)
                    if model.last_feasible_solution is None or model.incumbent_solution is None:
                        raise TypeError("Should not be None at this point.")
                    branching_constraints_efficacy_checker.check(
                        model.last_feasible_solution.variable_values, is_optimal_within_bounds=True
                    )
                    rhs = 1

                elif status_code == GRB.TIME_LIMIT:
                    if model.solution_count == 0:
                        break

                    model.store_solution()
                    model.pop_branching_constraints_stack()
                    model.prohibit_last_solution()
                    if model.last_feasible_solution is None or model.incumbent_solution is None:
                        raise TypeError("Should not be None at this point.")
                    branching_constraints_efficacy_checker.check(
                        model.last_feasible_solution.variable_values,
                        is_optimal_within_bounds=False,
                    )
                    rhs = 1

            if model.last_feasible_solution_better_than_incumbent():
                model.store_last_feasible_solution_as_incumbent()
                k_cur = k_step
            else:
                k_cur += k_step

            model.drop_all_branching_constraints()
            model.set_branching_var_values_for_shake()
            while time() - start_time < total_time_limit:
                model.add_shaking_constraints(k_cur, k_step)
                model.eliminate_cutoff()
                time_limit = total_time_limit - (time() - start_time)
                model.set_time_limit(max(0, time_limit))
                model.set_solution_limit(1)
                model.optimize()
                model.remove_shaking_constraints()

                if model.status == GRB.INFEASIBLE:
                    k_cur += k_step
                else:
                    model.store_solution()
                    break

        model.recover_to_best_solution_at_end()
        self.best_model = model
        self._post_processing()
        return self.best_model

    def run_vns_with_lb(
        self,
        total_time_limit: int | float = 60,
        node_time_limit: int | float = 5,
        k_min_perc: int | float = 20,
        k_step_perc: int | float = 20,
        l_min_perc: int | float = 10,
        l_step_perc: int | float = 10,
        initial_patience: float | int = 3,
        shake_patience: float | int = 2,
        drop_branching_constrs_before_shake: bool = False,
    ):
        k_min, l_min, k_step, l_step, k_max, l_max = self._absolute_branching_parameters(
            k_min_perc,
            l_min_perc,
            k_step_perc,
            l_step_perc,
        )

        model = ConstrainedModel(
            config=self.config,
            derived=self.derived,
        )

        start_time = time()
        k_cur = k_min

        time_limit = total_time_limit - (time() - start_time)
        model.set_time_limit(max(0, time_limit))
        model.optimize_while_momentum(patience=initial_patience)
        print(f"\nTIME ELAPSED: {time() - start_time}\n")
        model.store_solution()
        model.store_last_feasible_solution_as_incumbent()

        while self._time_not_over(start_time, total_time_limit):
            rhs = l_min
            while self._time_not_over(start_time, total_time_limit):
                if rhs > l_max:
                    break
                model.set_branching_var_values_for_vnd()
                model.add_bounding_branching_constraint(rhs)
                time_limit = min(node_time_limit, total_time_limit - (time() - start_time))
                model.set_time_limit(max(0, time_limit))
                model.set_cutoff()
                model.optimize()
                print(f"\nTIME ELAPSED: {time() - start_time}\n")
                model.pop_branching_constraints_stack()
                status_code = model.status
                if status_code in (GRB.INFEASIBLE, GRB.CUTOFF):
                    if rhs > l_min:
                        model.pop_branching_constraints_stack()
                    model.add_excluding_branching_constraint(rhs)
                    rhs += l_step

                elif status_code == GRB.OPTIMAL:
                    model.store_solution()
                    if rhs > l_min:
                        model.pop_branching_constraints_stack()
                    model.add_excluding_branching_constraint(rhs)
                    rhs = l_min

                elif status_code == GRB.TIME_LIMIT:
                    if model.solution_count == 0:
                        break

                    model.store_solution()
                    rhs = l_min

            if model.last_feasible_solution_better_than_incumbent():
                model.store_last_feasible_solution_as_incumbent()
                k_cur = k_min
            else:
                k_cur += k_step
                if k_cur > k_max:
                    k_cur = k_min
            if drop_branching_constrs_before_shake:
                model.drop_all_branching_constraints()

            model.set_branching_var_values_for_shake()
            while self._time_not_over(start_time, total_time_limit):
                model.add_shaking_constraints(k_cur, k_step)
                model.eliminate_cutoff()
                time_limit = total_time_limit - (time() - start_time)
                model.set_time_limit(max(0, time_limit))
                model.optimize_while_momentum(patience=shake_patience)
                print(f"\nTIME ELAPSED: {time() - start_time}\n")
                model.remove_shaking_constraints()
                if model.status == GRB.INFEASIBLE:
                    k_cur += k_step
                    if k_cur > k_max:
                        k_cur = k_min
                else:
                    model.store_solution()
                    break

        model.recover_to_best_solution_at_end()
        self.best_model = model
        self._post_processing()
        return self.best_model

    def _time_not_over(self, start_time: float, total_time_limit: int | float):
        return time() - start_time < total_time_limit

    def _absolute_branching_parameters(
        self,
        k_min_perc: int | float,
        l_min_perc: int | float,
        k_step_perc: int | float,
        l_step_perc: int | float,
    ) -> list[int]:
        branching_params_percentages = [
            k_min_perc,
            l_min_perc,
            k_step_perc,
            l_step_perc,
        ]
        if any(not 0 < param <= 100 for param in branching_params_percentages):
            raise ValueError("Percentages must be greater than 0 not greater than 100.")

        # The following k_max and l_max are set as they are due to the following: If a student
        # switches the group the variable for him/her in his current group changes its value by one
        # and the value of the variable for him/her in his future group changes by one. The change
        # in variables indicating whether groups are opened or closed is likely to be far smaller,
        # since e.g. groups have to have consecutive IDs.

        k_max = self.config.number_of_students * 2
        l_max = k_max
        branching_params = list(
            round(percentage / 100 * k_max) for percentage in branching_params_percentages
        )
        if any(param == 0 for param in branching_params):
            raise ValueError("An absolute branching parameter is zero due to rounding.")
        branching_params += [k_max, l_max]
        return branching_params

    def _post_processing(self):
        if self.best_model is None:
            raise TypeError("No postprocessing possible if best model is None.")
        retriever = SolutionInformationRetriever(
            config=self.config,
            derived=self.derived,
            constrained_model=self.best_model,
        )
        viewer = SolutionViewer(derived=self.derived, retriever=retriever)
        checker = SolutionChecker(
            config=self.config,
            derived=self.derived,
            constrained_model=self.best_model,
            retriever=retriever,
        )
        self.best_solution = Solution(
            config=self.config,
            derived=self.derived,
            constrained_model=self.best_model,
            retriever=retriever,
            viewer=viewer,
            checker=checker,
        )
        if checker.is_correct:
            print("IS CORRECT")
        else:
            print("IS INCORRECT")


if __name__ == "__main__":
    vns = VariableNeighborhoodSearch(5, 50, 4, 2, 100)
    vns.run_vns_with_lb(total_time_limit=10_000, node_time_limit=5)
