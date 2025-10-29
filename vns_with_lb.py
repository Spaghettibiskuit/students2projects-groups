"""A class that controls all components which enable VNS with linear branching."""

from time import time

from gurobipy import GRB

from branching_constraints_efficacy_checker import BranchingConstraintsEfficacyChecker
from configuration import Configuration
from constrained_model import ConstrainedModel
from derived_modeling_data import DerivedModelingData
from model_components import LinExpressions, Variables
from solution import Solution
from solution_checker import SolutionChecker
from solution_info_retriever import SolutionInformationRetriever
from solution_viewer import SolutionViewer


class VariableNeighborhoodSearch:
    """Controls all components which enable VNS with linear branching."""

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
        active_model = ConstrainedModel.initial_model(
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
        active_model = ConstrainedModel.initial_model(
            config=self.config,
            derived=self.derived,
        )
        start_time = time()
        k_cur = k_step
        active_model.set_solution_limit(1)
        time_limit = total_time_limit - (time() - start_time)
        active_model.set_time_limit(max(0, time_limit))
        active_model.optimize()
        active_model.save_var_values()
        active_model.save_decision_var_values()
        self.best_model = active_model.copy()
        current_model = active_model.copy()
        active_model.set_cutoff(ascending=False)
        while time() - start_time < total_time_limit:
            rhs = 1
            active_model.eliminate_solution_limit()
            branching_constraints_efficacy_checker = BranchingConstraintsEfficacyChecker()
            while time() - start_time < total_time_limit:
                active_model.add_bounding_branching_constraint(rhs)
                time_limit = min(node_time_limit, total_time_limit - (time() - start_time))
                active_model.set_time_limit(max(0, time_limit))
                active_model.optimize()
                status_code = active_model.status
                if proven_infeasible := status_code in (GRB.INFEASIBLE, GRB.CUTOFF):
                    active_model.drop_latest_branching_constraint()
                    active_model.add_excluding_branching_constraint(rhs)
                    rhs += 1

                elif status_code == GRB.OPTIMAL:
                    active_model.save_var_values()
                    current_model = active_model.copy()
                    active_model.drop_latest_branching_constraint()
                    active_model.add_excluding_branching_constraint(rhs)
                    active_model.save_decision_var_values()
                    branching_constraints_efficacy_checker.check(
                        active_model.saved_var_values, is_optimal_within_bounds=True
                    )
                    rhs = 1

                elif status_code == GRB.TIME_LIMIT:
                    if active_model.solution_count == 0:
                        break

                    active_model.save_var_values()
                    current_model = active_model.copy()
                    active_model.drop_latest_branching_constraint()
                    active_model.prohibit_last_solution()
                    active_model.save_decision_var_values()
                    branching_constraints_efficacy_checker.check(
                        active_model.saved_var_values, is_optimal_within_bounds=False
                    )
                    rhs = 1

                if not proven_infeasible:
                    active_model.set_cutoff(ascending=False)

            if current_model.objective_value > self.best_model.objective_value:
                self.best_model = current_model.copy()
                k_cur = k_step
            else:
                k_cur += k_step

            active_model = self.best_model.copy()
            active_model.drop_all_branching_constraints()
            while time() - start_time < total_time_limit:
                active_model.add_shaking_constraints(k_cur, k_step)
                active_model.eliminate_cutoff()
                time_limit = total_time_limit - (time() - start_time)
                active_model.set_time_limit(max(0, time_limit))
                active_model.set_solution_limit(1)
                active_model.optimize()
                active_model.remove_shaking_constraints()
                if active_model.status == GRB.INFEASIBLE:
                    k_cur += k_step
                else:
                    active_model.save_var_values()
                    active_model.save_decision_var_values()
                    break

        self._post_processing()
        return self.best_model

    def run_vns_with_lb(
        self,
        total_time_limit: int | float = 10,
        node_time_limit: int | float = 0.5,
        k_min_perc: int | float = 1,
        k_step_perc: int | float = 1,
        l_min_perc: int | float = 3,
        l_step_perc: int | float = 3,
    ):
        k_min, k_step, l_min, l_step = self._absolute_branching_parameters(
            k_min_perc,
            k_step_perc,
            l_min_perc,
            l_step_perc,
        )

        active_model = ConstrainedModel.initial_model(
            config=self.config,
            derived=self.derived,
        )
        start_time = time()
        k_cur = k_min
        active_model.set_solution_limit(1)
        time_limit = total_time_limit - (time() - start_time)
        active_model.set_time_limit(max(0, time_limit))
        active_model.optimize()
        active_model.save_var_values()
        active_model.save_decision_var_values()
        self.best_model = active_model.copy()
        current_model = active_model.copy()
        active_model.set_cutoff()
        while self._time_not_over(start_time, total_time_limit):
            rhs = l_min
            active_model.eliminate_solution_limit()
            while self._time_not_over(start_time, total_time_limit):
                active_model.add_bounding_branching_constraint(rhs)
                time_limit = min(node_time_limit, total_time_limit - (time() - start_time))
                active_model.set_time_limit(max(0, time_limit))
                active_model.optimize()
                status_code = active_model.status
                if proven_infeasible := status_code in (GRB.INFEASIBLE, GRB.CUTOFF):
                    active_model.drop_latest_branching_constraint()
                    active_model.add_excluding_branching_constraint(rhs)
                    rhs += l_step

                elif status_code == GRB.OPTIMAL:
                    active_model.save_var_values()
                    current_model = active_model.copy()
                    active_model.drop_latest_branching_constraint()
                    active_model.add_excluding_branching_constraint(rhs)
                    active_model.save_decision_var_values()
                    rhs = l_min

                elif status_code == GRB.TIME_LIMIT:
                    if active_model.solution_count == 0:
                        break

                    active_model.save_var_values()
                    current_model = active_model.copy()
                    active_model.drop_latest_branching_constraint()
                    active_model.save_decision_var_values()
                    rhs = l_min

                if not proven_infeasible:
                    active_model.set_cutoff()

            if current_model.objective_value > self.best_model.objective_value:
                self.best_model = current_model.copy()
                k_cur = k_min
            else:
                k_cur += k_step

            if self._time_not_over(start_time, total_time_limit):
                active_model = self.best_model.copy()
                active_model.drop_all_branching_constraints()
            while self._time_not_over(start_time, total_time_limit):
                active_model.add_shaking_constraints(k_cur, k_step)
                active_model.eliminate_cutoff()
                time_limit = total_time_limit - (time() - start_time)
                active_model.set_time_limit(max(0, time_limit))
                active_model.set_solution_limit(1)
                active_model.optimize()
                active_model.remove_shaking_constraints()
                if active_model.status == GRB.INFEASIBLE:
                    k_cur += k_step
                else:
                    active_model.save_var_values()
                    active_model.save_decision_var_values()
                    break

        self._post_processing()
        return self.best_model

    def _time_not_over(self, start_time: float, total_time_limit: int | float):
        return time() - start_time < total_time_limit

    def _absolute_branching_parameters(
        self,
        k_min_perc: int | float,
        k_step_perc: int | float,
        l_min_perc: int | float,
        l_step_perc: int | float,
    ) -> list[int]:
        branching_params_percentages = [
            k_min_perc,
            k_step_perc,
            l_min_perc,
            l_step_perc,
        ]
        if any(not 0 < param <= 100 for param in branching_params_percentages):
            raise ValueError("Percentages must be greater than 0 not greater than 100.")
        num_decision_variables = len(self.derived.project_group_student_triples) + len(
            self.derived.project_group_pairs
        )
        branching_params = list(
            round(percentage / 100 * num_decision_variables)
            for percentage in branching_params_percentages
        )
        if any(param == 0 for param in branching_params):
            raise ValueError("An absolute branching parameter is zero due to rounding.")
        return branching_params

    def _post_processing(self):
        if self.best_model is None:
            raise TypeError("No postprocessing possible if best model is None.")
        variables = Variables.get(self.derived, self.best_model)
        lin_expressions = LinExpressions.get(self.config, self.derived, variables)
        retriever = SolutionInformationRetriever(
            config=self.config,
            derived=self.derived,
            constrained_model=self.best_model,
            variables=variables,
            lin_expressions=lin_expressions,
        )
        viewer = SolutionViewer(derived=self.derived, retriever=retriever)
        checker = SolutionChecker(
            config=self.config,
            derived=self.derived,
            variables=variables,
            lin_expressions=lin_expressions,
            retriever=retriever,
        )
        self.best_solution = Solution(
            config=self.config,
            derived=self.derived,
            variables=variables,
            lin_expressions=lin_expressions,
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
    vns.run_vns_with_lb(total_time_limit=10_000)
