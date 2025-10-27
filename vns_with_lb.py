"""A class that controls all components which enable VNS with linear branching."""

from time import time

from gurobipy import GRB

from configuration import Configuration
from constrained_model import ConstrainedModel
from derived_modeling_data import DerivedModelingData
from model_components import LinExpressions, Variables
from solution import Solution
from solution_checker import SolutionChecker
from solution_info_retriever import SolutionInformationRetriever
from solution_viewer import SolutionViewer
from uniqueness_checker import UniquenessChecker


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

    def run_vns_with_lb(
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
        self.best_model = active_model.copy()
        current_model = active_model.copy()
        active_model.set_cutoff()
        while time() - start_time < total_time_limit:
            continue_vnd = True
            rhs = 1
            active_model.eliminate_solution_limit()
            while continue_vnd and (time() - start_time < total_time_limit):
                uniqueness_checker = UniquenessChecker()
                active_model.add_bounding_branching_constraint(rhs)
                time_limit = min(node_time_limit, total_time_limit - (time() - start_time))
                active_model.set_time_limit(max(0, time_limit))
                active_model.optimize()
                status_code = active_model.status
                if proven_infeasible := status_code in (GRB.INFEASIBLE, GRB.CUTOFF):
                    # active_model.recover()
                    active_model.drop_latest_branching_constraint()
                    active_model.add_excluding_branching_constraint(rhs)
                    rhs += 1

                elif status_code == GRB.OPTIMAL:
                    current_model = active_model.copy()
                    active_model.drop_latest_branching_constraint()
                    active_model.add_excluding_branching_constraint(rhs)
                    active_model.save_var_values()
                    uniqueness_checker.is_new(active_model.saved_var_values)
                    rhs = 1

                elif status_code == GRB.TIME_LIMIT:
                    if active_model.solution_count == 0:
                        continue_vnd = False
                        continue

                    current_model = active_model.copy()
                    active_model.drop_latest_branching_constraint()
                    active_model.prohibit_last_solution()
                    active_model.save_var_values()
                    uniqueness_checker.is_new(active_model.saved_var_values)
                    rhs = 1

                if not proven_infeasible:
                    active_model.set_cutoff()

            if current_model.objective_value > self.best_model.objective_value:
                self.best_model = current_model.copy()
                k_cur = k_step
            else:
                k_cur += k_step

            active_model = self.best_model.copy()
            active_model.drop_all_branching_constraints()
            continue_shake = True
            while continue_shake and (time() - start_time < total_time_limit):
                active_model.add_shaking_constraints(k_cur, k_step)
                active_model.eliminate_cutoff()
                time_limit = total_time_limit - (time() - start_time)
                active_model.set_time_limit(max(0, time_limit))
                active_model.set_solution_limit(1)
                active_model.optimize()
                if continue_shake := active_model.status == GRB.INFEASIBLE:
                    k_cur += k_step
                else:
                    active_model.save_var_values()
                active_model.remove_shaking_constraints()

        self._post_processing()
        return self.best_model

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
    vns = VariableNeighborhoodSearch(3, 30, 0, 2, 3)
    vns.run_vns_with_lb(10, 0.5, 2)
