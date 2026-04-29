import numpy as np
import pandas as pd
from json import loads
from random import random, sample
from math import log
from sympy import exp, sympify
from typing import Dict, List, Tuple, Union
from time import time

from evo_tools.bin_gray import range_of_numbers_binary_and_gray, \
  get_float_from_custom_representation, get_binary_from_custom_representation, \
  get_gray_from_custom_representation
from evo_tools.crossover import apply_crossover, validate_crossover_method
from evo_tools.helpers import sub_strings_by_array
from evo_tools.models import Individual, SubPopulation
from evo_tools.mutation import mutate_individual, validate_mutation_method
from evo_tools.phenotype import build_individual, decode_binary_segments, \
  validate_binaries_in_range
from evo_tools.scoring import assign_scores, sort_population_by_score
from evo_tools.selection import select_parents, validate_parent_selection_method

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

    for rng in ranges:
      sub_population_range, bits = range_of_numbers_binary_and_gray(
        rng,
        self._precision,
        self._sample_size
      )
      self._sub_populations.append(
        SubPopulation(rng, sub_population_range, bits)
      )
    self._max_sample_size = len(self._sub_populations[0].numbers)

    for sub_population in self._sub_populations:
      aux = len(sub_population.numbers)

      if aux < self._max_sample_size:
        self._max_sample_size = aux

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
      if self._print:
        print('\nInitial population:\n')
        print(self._initial_population)

      return self._initial_population.copy()
    else:
      samples: List[Tuple[List[str], int]] = []

      for sub_population in self._sub_populations:
        samples.append((
          sample(sub_population.numbers, self._sample_size),
          sub_population.bits
        ))

      first_sample, _ = samples[0]

      for i, _ in enumerate(first_sample):
        binary = ''
        gray = ''
        numbers = '['
        bits: List[int] = []

        for j, s in enumerate(samples):
          current_sample, current_bits = s
          bits.append(current_bits)
          binary += get_binary_from_custom_representation(current_sample[i])
          gray += get_gray_from_custom_representation(current_sample[i])

          if j != len(samples) - 1:
            numbers += f'{get_float_from_custom_representation(current_sample[i])}, '
          else:
            numbers += f'{get_float_from_custom_representation(current_sample[i])}]'

        self._initial_population.append(
          build_individual(
            binary,
            gray,
            bits,
            self._sub_populations,
            self._precision,
            self._parsed_function,
            self._variables_array,
            numbers
          )
        )

      self._current_population = self._initial_population.copy()

      if self._print:
        print('\nInitial population:\n')
        print(self._initial_population)
        print()

      return self._current_population.copy()

  def _get_current_population(self) -> List[Individual]:
    """
    Returns a copy of the current Population data.

    Returns:
      List[Individual]
    """
    return self._current_population.copy()

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

    self._fitness(self._current_population, minimize)
    sort_population_by_score(self._current_population)

  def _select(
    self,
    individuals: List[Individual],
    minimize: bool,
    sample_size: int,
    mutation_method: str
  ) -> None:
    mutated_individuals = self._mutation(individuals, mutation_method)
    self._fitness(mutated_individuals, minimize)
    sort_population_by_score(mutated_individuals)

    # Calculate the mean and std of the population before the selection
    current_population_score = np.array([
      x.get_score() for x in self._current_population
    ])
    score_std_before_selection = float(np.std(current_population_score))
    score_mean_before_selection = float(np.mean(current_population_score))

    self._update_current_population(
      self._current_population[
        :len(self._current_population) - len(mutated_individuals)
      ] + mutated_individuals[:sample_size],
      minimize
    )

    # Calculate the mean of the population after the selection
    current_population_score = np.array([
      x.get_score() for x in self._current_population
    ])
    score_mean_after_selection = float(np.mean(current_population_score))
    self._selection_strength = (
      float(abs(
        (score_mean_after_selection - score_mean_before_selection) / score_std_before_selection
      )) if score_std_before_selection > 0 else 0.0
    )

    self._best_individual = self._current_population[0]

  def _mutation(
    self,
    children: List[Individual],
    mutation_method = 'one_point'
  ) -> List[Individual]:
    if self._print:
      print(f'\nPopulation children before mutation: {children}\n')
      print()

    mutated_children: List[Individual] = []

    for child in children:
      mutated_child = child

      if random() < self._mutation_rate:
        if (self._print):
          print(f'  Mutation for child: {child}\n')

        valid_mutation = mutate_individual(
          child,
          mutation_method,
          self._sub_populations,
          self._precision,
          self._parsed_function,
          self._variables_array
        )

        if valid_mutation is not None:
          mutated_child = valid_mutation

          if (self._print):
            print(f'  Mutation for child completed: {mutated_child}\n')

      mutated_children.append(mutated_child)

    if self._print:
      print(f'\nPopulation children after mutation: {mutated_children}\n')
      print()

    return mutated_children

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
      chromosome = individual.get_binary()

      if (self._print):
        print(f'Chromosome {i}: {chromosome}')

      gens, fens = self._decode_individual(individual)

      if (self._print):
        print(f'  gens: {gens}')
        print(f'  fens: {fens}')

      function: exp = self._parsed_function

      if len(gens) == len(fens):
        # Evaluate the given function variable per variable
        for i, v in enumerate(self._variables_array):
          function = function.subs(v, fens[i])  # type: ignore

        objective_value = float(function)
        function_evaluations.append(objective_value)
        individual.set_objective_value(objective_value)

        if self._print:
          print(f'  fitness: {objective_value}\n')
      elif self._print:
        print(f'  fitness: Fail\n')

    if len(function_evaluations) == 0:
      return

    assign_scores(population_sample, function_evaluations, minimize)

  def _select_parents(
    self,
    seed: float,
    parent_selection_method,
    minimize: bool
  ):
    return select_parents(
      self._current_population,
      seed,
      parent_selection_method,
      minimize
    )

  def _do_crossover_using_a_method(
    self,
    seed: float,
    crossover_method,
    parent_selection_method,
    minimize: bool
  ):
    parents = self._select_parents(
      seed,
      parent_selection_method,
      minimize
    )

    if len(parents) == 0:
      return []

    return apply_crossover(
      parents,
      crossover_method,
      self._crossover_rate,
      self._sub_populations,
      self._precision,
      self._parsed_function,
      self._variables_array
    )

  def _population_fitness_average(self) -> float:
    return float(np.mean(
      np.array(
        [individual.get_fitness() for individual in self._current_population] # type: ignore
      )
    ))

  def _decode_individual(
    self,
    individual: Individual
  ) -> Tuple[List[str], List[float]]:
    gens = sub_strings_by_array(
      individual.get_binary(),
      individual.get_bits()
    )
    fens: List[float] = []

    if validate_binaries_in_range([gens], self._sub_populations, self._precision):
      _, fens = decode_binary_segments(
        gens,
        self._sub_populations,
        self._precision
      )

    return gens, fens

  def _build_solution(
    self,
    floats: List[float]
  ) -> Tuple[Dict[str, float], exp]:
    if len(floats) == 0:
      raise Exception('Something went wrong')

    function = self._parsed_function
    solution: Dict[str, float] = {}

    for i, v in enumerate(self._variables_array):
      function = function.subs(v, floats[i])
      solution[v] = floats[i]

    return solution, function

  def _print_iteration_summary(
    self,
    current_iteration: int,
    selection_strength: float,
    elapsed_time: float
  ) -> None:
    print(
      f'\n{current_iteration}º iteration.\nBest individual: {self._best_individual}.\nSelection strength: {selection_strength}.\nTime elapsed: {elapsed_time}s.'
    )
    df = pd.DataFrame(loads(str(self._current_population)))
    print(df, end = '\n\n')

  def _print_final_summary(
    self,
    current_iteration: int,
    fitness_avg_list: List[float],
    solution: Dict[str, float],
    function
  ) -> None:
    print(
      f'\n\nFinally:\n{current_iteration}º iteration.\nBest individual: {self._best_individual}.\nSelection strength: {self._selection_strength}.'
    )
    print('Solution:')
    print(f'  Variables: {solution}')
    print(f'  Evaluation: {function}')
    print(f'  Fitness average: {fitness_avg_list}')

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

    self._select_initial_population()
    start = time()
    self._fitness(self._current_population, MINIMIZE)
    sort_population_by_score(self._current_population)
    self._best_individual = self._current_population[0]
    current_iteration = 1
    scores: List[float] = []
    fitness_avg_list: List[float] = [self._population_fitness_average()]
    end = time()

    if PRINT:
      self._print_iteration_summary(
        current_iteration,
        self._selection_strength,
        end - start
      )

    for i in range(ITERATIONS - 1):
      start = time()
      current_iteration += 1
      children = self._do_crossover_using_a_method(
        self._sample_size * SEED,
        CROSSOVER_METHOD,
        PARENT_SELECTION_METHOD,
        MINIMIZE
      )
      self._select(children, MINIMIZE, self._sample_size, MUTATION_METHOD)
      scores.append(self._best_individual.get_score())
      fitness_avg_list.append(self._population_fitness_average())
      end = time()

      if self._selection_strength <= 1e-4:
        break

      if PRINT:
        self._print_iteration_summary(
          current_iteration,
          self._selection_strength,
          end - start
        )

    _, floats = self._decode_individual(self._best_individual)
    solution, function = self._build_solution(floats)

    if PRINT:
      self._print_final_summary(
        current_iteration,
        fitness_avg_list,
        solution,
        function
      )

    return scores, solution, function, fitness_avg_list
