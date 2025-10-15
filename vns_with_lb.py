"""A class that controls all components which enable VNS with linear branching."""

from configuration import Configuration
from constrained_model import ConstrainedModel
from derived_modeling_data import DerivedModelingData
from model_components import LinExpressions, Variables
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
        self.config = Configuration(
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
        self.variables: Variables | None = None
        self.lin_expressions: LinExpressions | None = None
        self.retriever: SolutionInformationRetriever | None = None
        self.viewer: SolutionViewer | None = None

    def solve_exactly(self) -> ConstrainedModel:
        self.initial_model.model.optimize()
        self.best_model = self.initial_model
        self.variables = Variables.get(self.derived, self.best_model)
        self.lin_expressions = LinExpressions.get(self.config, self.derived, self.variables)
        self.retriever = SolutionInformationRetriever(
            config=self.config,
            derived=self.derived,
            constrained_model=self.best_model,
            variables=self.variables,
            lin_expressions=self.lin_expressions,
        )
        self.viewer = SolutionViewer(derived=self.derived, retriever=self.retriever)
        return self.best_model
