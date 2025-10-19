"""A class that controls all components which enable VNS with linear branching."""

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
        self.derived = DerivedModelingData.get(self.config)
        self.initial_model = ConstrainedModel.initial_model(
            config=self.config,
            derived=self.derived,
        )
        self.best_model = None
        self.best_solution = None

    def gurobi_alone(self, time_limit: int | float = float("inf")) -> ConstrainedModel:
        self.initial_model.model.Params.TimeLimit = time_limit
        self.initial_model.model.optimize()

        self.best_model = self.initial_model

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
            config=self.config, derived=self.derived, variables=variables, retriever=retriever
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
        if checker.is_valid():
            print("IS VALID")
        return self.best_model
