import itertools
from time import time

from gurobipy import GRB

from configuration import Configuration
from constrained_model import ConstrainedModel
from derived_modeling_data import DerivedModelingData
from reduced_model import ReducedModel
from solution import Solution
from solution_checker import SolutionChecker
from solution_info_retriever import SolutionInformationRetriever
from solution_viewer import SolutionViewer
from thin_wrappers import (
    ConstrainedModelInitializer,
    GurobiDuck,
    ReducedModelInitializer,
)


class VariableNeighborhoodSearch:

    def __init__(
        self,
        number_of_projects: int,
        number_of_students: int,
        instance_index: int,
        reward_mutual_pair: int = 2,
        penalty_unassigned: int = 3,
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

    def gurobi_alone(self, time_limit: int | float = float("inf")) -> list[dict[str, int | float]]:
        model = GurobiDuck(
            config=self.config,
            derived=self.derived,
        )
        model.set_time_limit(time_limit)
        model.optimize()

        self.best_model = model
        self._post_processing()
        return model.solution_summaries

    def run_vns_with_lb(
        self,
        total_time_limit: int | float = 60,
        k_min_perc: int | float = 20,
        k_step_perc: int | float = 20,
        l_min_perc: int | float = 10,
        l_step_perc: int | float = 10,
        initial_patience: float | int = 3,
        shake_patience: float | int = 2,
        min_optimization_patience: int | float = 2,
        step_optimization_patience: int | float = 0.5,
        drop_branching_constrs_before_shake: bool = False,
    ):
        k_min, l_min, k_step, l_step, k_max, l_max = self._absolute_branching_parameters(
            k_min_perc,
            l_min_perc,
            k_step_perc,
            l_step_perc,
        )

        initial_model = ConstrainedModelInitializer(config=self.config, derived=self.derived)
        start_time = initial_model.start_time

        k_cur = k_min - k_step  # lets shake begin at k_min even if no better sol found during VND

        time_limit = max(0, total_time_limit - (time() - start_time))
        initial_model.set_time_limit(time_limit)
        initial_model.optimize(patience=initial_patience)

        model = ConstrainedModel.get(initial_model)
        model.make_current_solution_best_solution()

        while not self._time_over(start_time, total_time_limit):
            rhs = l_min
            patience = min_optimization_patience

            while not self._time_over(start_time, total_time_limit):
                if rhs > l_max:
                    break
                model.set_branching_var_values_for_vnd()
                model.add_bounding_branching_constraint(rhs)
                time_limit = max(0, total_time_limit - (time() - start_time))
                model.set_time_limit(time_limit)
                model.set_cutoff()
                model.optimize(patience)

                model.pop_branching_constraints_stack()
                status_code = model.status
                if status_code in (GRB.INFEASIBLE, GRB.CUTOFF):
                    if rhs > l_min:  # Kommentieren
                        model.pop_branching_constraints_stack()
                    model.add_excluding_branching_constraint(rhs)
                    rhs += l_step
                    patience += step_optimization_patience

                elif status_code == GRB.OPTIMAL:
                    model.store_solution()
                    if rhs > l_min:
                        model.pop_branching_constraints_stack()
                    model.add_excluding_branching_constraint(rhs)
                    rhs = l_min
                    patience = min_optimization_patience

                elif status_code == GRB.TIME_LIMIT:
                    if model.solution_count == 0:
                        break

                    model.store_solution()
                    rhs = l_min
                    patience = min_optimization_patience

            if model.new_best_found():
                model.make_current_solution_best_solution()
                k_cur = k_min
            else:
                k_cur += k_step
                if k_cur > k_max:
                    k_cur = k_min
            if drop_branching_constrs_before_shake:
                model.drop_all_branching_constraints()

            model.set_branching_var_values_for_shake()
            while not self._time_over(start_time, total_time_limit):
                model.add_shaking_constraints(k_cur, k_step)
                model.eliminate_cutoff()
                time_limit = max(0, total_time_limit - (time() - start_time))
                model.set_time_limit(time_limit)
                model.optimize(patience=shake_patience, shake=True)

                model.remove_shaking_constraints()
                if model.status == GRB.INFEASIBLE:
                    k_cur += k_step
                    if k_cur > k_max:
                        k_cur = k_min
                else:
                    model.store_solution()
                    break

        model.recover_to_best_found()
        self.best_model = model
        self._post_processing()
        return model.solution_summaries

    def run_vns_with_var_fixing(
        self,
        total_time_limit: int | float = 60,
        min_num_zones: int = 4,
        step_num_zones: int = 1,
        max_num_zones: int = 6,
        max_iterations_per_num_zones: int = 20,
        min_shake_perc: int = 10,
        step_shake_perc: int = 10,
        max_shake_perc: int = 50,
        initial_patience: int | float = 3,
        shake_patience: int | float = 2,
        min_optimization_patience: int | float = 1,
        step_optimization_patience: int | float = 1,
    ):
        min_shake, step_shake, max_shake = self._absolute_fixing_parameters(
            (min_shake_perc, step_shake_perc, max_shake_perc)
        )

        k = min_shake - step_shake

        initial_model = ReducedModelInitializer(self.config, self.derived)
        start_time = initial_model.start_time

        time_limit = max(0, total_time_limit - (time() - start_time))
        initial_model.set_time_limit(time_limit)
        initial_model.optimize(initial_patience)

        model = ReducedModel.get(initial_model)
        model.set_cutoff()

        while not self._time_over(start_time, total_time_limit):
            current_num_zones = max_num_zones
            iterations_current_num_zones = 0

            free_zones_pairs = itertools.combinations(range(current_num_zones), 2)
            new_pairs = False

            patience = min_optimization_patience

            while not self._time_over(start_time, total_time_limit):
                if new_pairs:
                    free_zones_pairs = itertools.combinations(range(current_num_zones), 2)
                    new_pairs = False
                    iterations_current_num_zones = 0

                if iterations_current_num_zones > max_iterations_per_num_zones:
                    if current_num_zones == min_num_zones:
                        break
                    new_pairs = True
                    current_num_zones = max(current_num_zones - step_num_zones, min_num_zones)
                    patience += step_optimization_patience
                    continue

                if (free_zones_pair := next(free_zones_pairs, None)) is None:
                    if current_num_zones == min_num_zones:
                        break

                    new_pairs = True
                    current_num_zones = max(current_num_zones - step_num_zones, min_num_zones)
                    patience += step_optimization_patience
                    continue

                iterations_current_num_zones += 1
                model.fix_rest(*free_zones_pair, current_num_zones)

                time_limit = max(0, total_time_limit - (time() - start_time))
                model.set_time_limit(time_limit)
                model.optimize(patience)

                if model.solution_count == 0:
                    continue

                model.store_solution()
                model.set_cutoff()

                new_pairs = True
                current_num_zones = max_num_zones
                patience = min_optimization_patience

            if model.new_best_found():
                model.make_current_solution_best_solution()
                k = min_shake
            elif k == max_shake:
                k = min_shake
                model.increment_random_seed()
                model.delete_zoning_rules()
            else:
                k = min(k + step_shake, max_shake)

            model.make_best_solution_current_solution()

            model.eliminate_cutoff()
            model.force_k_worst_to_change(k)

            time_limit = max(0, total_time_limit - (time() - start_time))
            model.set_time_limit(time_limit)
            model.optimize(shake_patience, shake=True)

            if model.status == GRB.TIME_LIMIT:
                break

            model.store_solution()
            if model.new_best_found():
                model.make_current_solution_best_solution()
                k = min_shake - step_shake
            model.set_cutoff()

        model.recover_to_best_found()
        self.best_model = model
        self._post_processing()
        return model.solution_summaries

    def _time_over(self, start_time: float, total_time_limit: int | float):
        return time() - start_time > total_time_limit

    def _absolute_fixing_parameters(self, shake_percentages: tuple[int, ...]) -> tuple[int, ...]:
        if any(not 0 < param <= 100 for param in shake_percentages):
            raise ValueError("Percentages must be greater than 0 not greater than 100.")
        shake_params = tuple(
            round(percentage / 100 * self.config.number_of_students)
            for percentage in shake_percentages
        )
        if any(param == 0 for param in shake_params):
            raise ValueError("An absolute branching parameter is zero due to rounding.")
        return shake_params

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
            wrapped_model=self.best_model,
        )
        viewer = SolutionViewer(derived=self.derived, retriever=retriever)
        checker = SolutionChecker(
            config=self.config,
            derived=self.derived,
            wrapped_model=self.best_model,
            retriever=retriever,
        )
        self.best_solution = Solution(
            config=self.config,
            derived=self.derived,
            wrapped_model=self.best_model,
            retriever=retriever,
            viewer=viewer,
            checker=checker,
        )
        if checker.is_correct:
            print("IS CORRECT")
        else:
            print("IS INCORRECT")
        self.best_model.solution_summaries.append({"is_correct": int(checker.is_correct)})


if __name__ == "__main__":
    vns = VariableNeighborhoodSearch(5, 50, 4, 2, 3)
    vns.run_vns_with_lb(total_time_limit=10_000)
