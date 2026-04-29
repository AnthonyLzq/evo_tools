from sympy import exp, lambdify, sympify
from typing import Dict, List, Tuple, Union

from evo_tools.canonical import finalize_canonical_result, \
  run_canonical_algorithm, \
  validate_canonical_methods
from evo_tools.domain import build_sub_populations, validate_variable_count
from evo_tools.models import Individual, SubPopulation

# ParentSelectionMethods = Literal['fitness_proportionate', 'roulette', 'tournament']
# CrossoverMethods = Literal['one_point', 'two_points', 'uniform']
# MutationMethods = Literal['one_point', 'two_points', 'flipping']

class Population():
  """
  A class to represent a population
  --

  It is conformed by several PopulationMembers.

  Attributes
  --

  _sub_populations: List[:class:`SubPopulation`]
    A list of every SubPopulation from the Population, which is defined by
    the range.

  _precision: Union[float, int]
    A float or number value to decision how precise the ranges must but.

    If an int is passed, it must 1 and it will working with natural values
    in the range (or a least will try to).

    If a float is passed, it must be a decimal fraction, something lik 0.1, 0.01, etc.

  _crossover_rate: float
    Probability to crossover children.

  _mutation_rate: float
    Probability to mutate children.

  _variables_array: List[str]
    Parsed variable names used across the extracted canonical helpers.

  _parsed_function: exp
    Sympified objective function reused across initialization and evolution.

  _objective_function:
    Numeric callable compiled once for repeated per-individual evaluation.

  _print: bool = False
    Whether or not should print the output in the methods.

  _current_population: List[Individual]
    Population selected that will change in every iteration of the canonical
    algorithm.

  _initial_population: List[Individual]
    First population selected by the canonical algorithm.

  _best_individual: Individual
    Individual with the highest score in the current population. That means it
    is the closest to the actual solution.
  """

  def __init__(
    self,
    ranges: List[Tuple[Union[float, int], Union[float, int]]],
    precision: Union[float, int],
    crossover_rate: float,
    mutation_rate: float,
    variables: str,
    function: exp,
    _print: bool = False,
    sample_size: int = 80
  ) -> None:
    """
    Constructor to initialize a Population

    Args:
      ranges (List[Tuple[Union[float, int], Union[float, int]]])
        A list of the ranges that are going to be used to define the Population.
        It represents the domain of every variable in the equation to be
        minimized or maximized.

      precision (Union[float, int]):
        A decimal fraction (0.1, 0.01, etc.) or 1 if Natural numbers will be used.

      crossover_rate (float):
        A decimal value that indicates the probability of parents crossover.

      mutation_rate (float):
        A decimal value that indicates the probability of children mutation.

      variables (str):
        String with the variables separated by blanks. For example 'x y z'.

      function (exp):
        An expression created with sympy with the given variables.

      _print (bool, optional):
        Whether or not the output should be printed. Defaults to False.

    Raises:
      Exception: when there wasn't any range for the Population or the number of
      variables doesn't match the number of ranges.
    """
    if len(ranges) == 0:
      raise Exception('At least one range is required')

    self._sub_populations: List[SubPopulation] = []
    self._precision = precision
    self._crossover_rate = crossover_rate
    self._mutation_rate = mutation_rate
    self._variables_array = variables.split()
    self._parsed_function = sympify(function)
    self._objective_function = lambdify(
      self._variables_array,
      self._parsed_function,
      modules = 'math'
    )
    self._print = _print
    self._current_population: List[Individual] = []
    self._initial_population: List[Individual] = []
    self._best_individual: Individual
    self._selection_strength: float = 0
    self._sample_size = sample_size

    self._sub_populations = build_sub_populations(
      ranges,
      self._precision,
      self._sample_size
    )
    validate_variable_count(self._variables_array, self._sub_populations)

  def canonical_algorithm(
    self,
    ITERATIONS = 100,
    MINIMIZE = True,
    SEED = 1.1,
    PRINT = False,
    PARENT_SELECTION_METHOD = 'fitness_proportionate',
    CROSSOVER_METHOD = 'one_point',
    MUTATION_METHOD = 'one_point'
  ) -> Tuple[List[float], Dict[str, float], exp, List[float]]:
    """
    Canonical algorithm that follows the following steps:
    1. select initial population
    2. fitness the initial population
    3. select the best individual
    4. a loop:
      4.1. crossover
      4.2. selection
      4.3. validate error, if error <= 1e-3, break, else continue
    5. calculates the solution and return it

    Args:
      SAMPLE_SIZE (int): sample size of the population to work with.
      ITERATIONS (int, optional): number of iterations for the algorithm.
      Defaults to 200.
      MINIMIZE (bool, optional): a boolean that indicates if the problem
      is it a minimization or maximization problem. Defaults to True.
      SEED (float, optional): fraction of the sample size to fix the maximum
      number of parents. Defaults to 1.8.
      PRINT (bool, optional): a boolean that indicates if the output should be
      printed. Defaults to False.
      PARENT_SELECTION_METHOD (ParentSelectionMethods, optional): a method to
      select the parents. Defaults to 'fitness_proportionate'.
      CHILDREN_GENERATION_METHOD (CrossoverMethods, optional): a method to
      crossover. Defaults to 'one_point'.
      MUTATION_METHOD (MutationMethods, optional): a method to mutate. Default
      to 'one_point'.

    Raises:
      Exception: when a generation has a individual that is outside from all the
      given intervals.

    Returns:
      Tuple[List[float], Dict[str, float], exp, fitness_avg_list]: A tuple that
      contains the list of historical scores from each generation, a Dict with
      the solution for each given variable, the result calculated for the
      obtained solution and the average of the fitness per each generation.
    """
    validate_canonical_methods(
      PARENT_SELECTION_METHOD,
      CROSSOVER_METHOD,
      MUTATION_METHOD
    )

    self._initial_population, self._current_population, self._best_individual, \
      self._selection_strength, current_iteration, scores, fitness_avg_list = \
      run_canonical_algorithm(
        self._initial_population,
        self._sample_size,
        self._sub_populations,
        self._precision,
        self._parsed_function,
        self._variables_array,
        self._objective_function,
        MINIMIZE,
        ITERATIONS,
        SEED,
        PRINT,
        PARENT_SELECTION_METHOD,
        CROSSOVER_METHOD,
        MUTATION_METHOD,
        self._crossover_rate,
        self._mutation_rate
      )

    solution, function = finalize_canonical_result(
      self._best_individual,
      self._parsed_function,
      self._variables_array,
      current_iteration,
      self._selection_strength,
      fitness_avg_list,
      PRINT
    )

    return scores, solution, function, fitness_avg_list
