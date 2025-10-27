class UniquenessChecker:

    def __init__(self):
        self.unique_solution_variables: set[tuple[int | float, ...]] = set()

    def is_new(self, solution_variables: tuple[int | float, ...]):
        if solution_variables in self.unique_solution_variables:
            raise ValueError("Solution already found before!")
        self.unique_solution_variables.add(solution_variables)
