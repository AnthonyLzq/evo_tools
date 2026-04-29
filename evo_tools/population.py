from random import sample
from math import log
from sympy import exp, sympify
from typing import Dict, List, Tuple, Union
from time import time

from evo_tools.bin_gray import range_of_numbers_binary_and_gray, \
  get_float_from_custom_representation, get_binary_from_custom_representation, \
  get_gray_from_custom_representation
from evo_tools.crossover import generate_children, validate_crossover_method
from evo_tools.models import Individual, SubPopulation
from evo_tools.mutation import mutate_children, validate_mutation_method
from evo_tools.phenotype import build_individual, build_solution, \
  decode_individual, evaluate_individual_objective
from evo_tools.reporting import print_final_summary, print_iteration_summary
from evo_tools.scoring import assign_scores, population_fitness_average, \
  population_score_stats, selection_strength, sort_population_by_score
from evo_tools.selection import validate_parent_selection_method

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

  _variables: str
    The variables to be used in the objective function, separated by spaces.

  _function: exp
    The function to be minimized of maximized.

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
    self._variables = variables
    self._function = function
    self._variables_array = self._variables.split()
    self._parsed_function = sympify(str(self._function))
    self._print = _print
    self._current_population: List[Individual] = []
    self._initial_population: List[Individual] = []
    self._best_individual: Individual
    self._selection_strength: float = 0
    self._sample_size = sample_size

    p10 = 1 if precision == 1 else pow(precision, -1)
    self._n_decimal_digits = int(round(log(p10, 10)))
    self._build_sub_populations(ranges)
    self._max_sample_size = self._resolve_max_sample_size()
    self._validate_variable_count()

  def _build_sub_populations(
    self,
    ranges: List[Tuple[Union[float, int], Union[float, int]]]
  ) -> None:
    for rng in ranges:
      sub_population_range, bits = range_of_numbers_binary_and_gray(
        rng,
        self._precision,
        self._sample_size
      )
      self._sub_populations.append(
        SubPopulation(rng, sub_population_range, bits)
      )

  def _resolve_max_sample_size(self) -> int:
    max_sample_size = len(self._sub_populations[0].numbers)

    for sub_population in self._sub_populations:
      sub_population_size = len(sub_population.numbers)

      if sub_population_size < max_sample_size:
        max_sample_size = sub_population_size

    return max_sample_size

  def _validate_variable_count(self) -> None:
    if (len(self._variables_array) != len(self._sub_populations)):
      raise Exception('Variables size does not match the number of ranges')

  def _select_initial_population(self) -> List[Individual]:
    """
    Method that selects the initial sample (randomly) of the Population.

    Args:
      sample_size: int
        Population sample size.

    Raises:
      Exception: When the required sample_size is bigger than the maximum sample
      size (the lowest range size from the domain).

    Returns:
      List[Individual]: A list of List[:class:`Individual`] which represents the
      initial population
    """
    if (self._sample_size > self._max_sample_size):
      raise Exception(
        f'Sample size too big, maximum is: {self._max_sample_size}'
      )

    if len(self._initial_population) > 0:
      self._print_initial_population()
      return self._initial_population.copy()

    samples = self._sample_sub_populations()
    first_sample, _ = samples[0]

    for i, _ in enumerate(first_sample):
      self._initial_population.append(
        self._build_initial_individual(samples, i)
      )

    self._current_population = self._initial_population.copy()
    self._print_initial_population()

    return self._current_population.copy()

  def _get_current_population(self) -> List[Individual]:
    """
    Returns a copy of the current Population data.

    Returns:
      List[Individual]
    """
    return self._current_population.copy()

  def _sample_sub_populations(self) -> List[Tuple[List[str], int]]:
    return [
      (
        sample(sub_population.numbers, self._sample_size),
        sub_population.bits
      )
      for sub_population in self._sub_populations
    ]

  def _build_initial_individual(
    self,
    samples: List[Tuple[List[str], int]],
    index: int
  ) -> Individual:
    selected_numbers = [current_sample[index] for current_sample, _ in samples]
    bits = [current_bits for _, current_bits in samples]
    binary = ''.join(
      get_binary_from_custom_representation(number)
      for number in selected_numbers
    )
    gray = ''.join(
      get_gray_from_custom_representation(number)
      for number in selected_numbers
    )
    numbers = '[' + ', '.join(
      get_float_from_custom_representation(number)
      for number in selected_numbers
    ) + ']'

    return build_individual(
      binary,
      gray,
      bits,
      self._sub_populations,
      self._precision,
      self._parsed_function,
      self._variables_array,
      numbers
    )

  def _print_initial_population(self) -> None:
    if self._print:
      print('\nInitial population:\n')
      print(self._initial_population)
      print()

  def _rank_population(
    self,
    population_sample: List[Individual],
    minimize: bool
  ) -> None:
    self._fitness(population_sample, minimize)
    sort_population_by_score(population_sample)

  def _update_current_population(
    self,
    new_population: List[Individual],
    minimize: bool
  ) -> None:
    """
    Method that updates the current sample, after crossover or mutation.

    Args:
      new_population (List[Individual])
    """
    if (len(new_population) > self._sample_size):
      self._current_population = new_population[:self._sample_size]
    else:
      self._current_population = new_population

    self._rank_population(self._current_population, minimize)
    self._refresh_best_individual()

  def _compose_next_population(
    self,
    mutated_individuals: List[Individual],
    sample_size: int
  ) -> List[Individual]:
    return self._current_population[
      :len(self._current_population) - len(mutated_individuals)
    ] + mutated_individuals[:sample_size]

  def _select_next_generation(
    self,
    individuals: List[Individual],
    minimize: bool,
    sample_size: int,
    mutation_method: str
  ) -> None:
    mutated_individuals = mutate_children(
      individuals,
      mutation_method,
      self._mutation_rate,
      self._sub_populations,
      self._precision,
      self._parsed_function,
      self._variables_array,
      self._print
    )
    self._rank_population(mutated_individuals, minimize)

    score_mean_before_selection, score_std_before_selection = population_score_stats(
      self._current_population
    )

    self._update_current_population(
      self._compose_next_population(mutated_individuals, sample_size),
      minimize
    )

    self._selection_strength = selection_strength(
      self._current_population,
      score_mean_before_selection,
      score_std_before_selection
    )

  def _fitness(self, population_sample: List[Individual], minimize: bool) -> None:
    """
    Method that calculates the genotype fitness of a given function for the
    current population.

    Args:
      population_sample (List[Individual]): a subset from the current population
      to calculate its fitness.
      minimize (bool, optional): a boolean that indicates if the problem
      is it a minimization or maximization problem. Defaults to True.
    """
    if len(population_sample) == 0:
      return

    function_evaluations: List[float] = []

    for i, individual in enumerate(population_sample):
      individual.set_score(0)
      individual.set_objective_value(None)
      objective_value = evaluate_individual_objective(
        individual,
        i,
        self._sub_populations,
        self._precision,
        self._parsed_function,
        self._variables_array,
        self._print
      )

      if objective_value is not None:
        function_evaluations.append(objective_value)
        individual.set_objective_value(objective_value)

    if len(function_evaluations) == 0:
      return

    assign_scores(population_sample, function_evaluations, minimize)

  def _refresh_best_individual(self) -> None:
    self._best_individual = self._current_population[0]

  def _initialize_canonical_state(
    self,
    minimize: bool
  ) -> Tuple[int, List[float], List[float]]:
    self._select_initial_population()
    self._rank_population(self._current_population, minimize)
    self._refresh_best_individual()

    return 1, [], [population_fitness_average(self._current_population)]

  def _run_canonical_iteration(
    self,
    seed: float,
    crossover_method,
    parent_selection_method,
    mutation_method: str,
    minimize: bool,
    scores: List[float],
    fitness_avg_list: List[float]
  ) -> None:
    children = generate_children(
      self._current_population,
      self._sample_size * seed,
      crossover_method,
      parent_selection_method,
      minimize,
      self._crossover_rate,
      self._sub_populations,
      self._precision,
      self._parsed_function,
      self._variables_array
    )
    self._select_next_generation(
      children,
      minimize,
      self._sample_size,
      mutation_method
    )
    scores.append(self._best_individual.get_score())
    fitness_avg_list.append(population_fitness_average(self._current_population))

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
    validate_parent_selection_method(PARENT_SELECTION_METHOD)
    validate_crossover_method(CROSSOVER_METHOD)
    validate_mutation_method(MUTATION_METHOD)

    start = time()
    current_iteration, scores, fitness_avg_list = self._initialize_canonical_state(
      MINIMIZE
    )
    end = time()

    if PRINT:
      print_iteration_summary(
        current_iteration,
        self._best_individual,
        self._selection_strength,
        end - start,
        self._current_population
      )

    for i in range(ITERATIONS - 1):
      start = time()
      current_iteration += 1
      self._run_canonical_iteration(
        SEED,
        CROSSOVER_METHOD,
        PARENT_SELECTION_METHOD,
        MUTATION_METHOD,
        MINIMIZE,
        scores,
        fitness_avg_list
      )
      end = time()

      if self._selection_strength <= 1e-4:
        break

      if PRINT:
        print_iteration_summary(
          current_iteration,
          self._best_individual,
          self._selection_strength,
          end - start,
          self._current_population
        )

    _, floats = decode_individual(
      self._best_individual,
      self._sub_populations,
      self._precision
    )
    solution, function = build_solution(
      floats,
      self._parsed_function,
      self._variables_array
    )

    if PRINT:
      print_final_summary(
        current_iteration,
        self._best_individual,
        self._selection_strength,
        fitness_avg_list,
        solution,
        function
      )

    return scores, solution, function, fitness_avg_list
