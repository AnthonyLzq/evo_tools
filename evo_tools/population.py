import numpy as np
import pandas as pd
from json import loads
from random import choice, random, sample
from math import log
from sympy import exp, sympify
from typing import Dict, List, Tuple, Union
from time import time

from evo_tools.bin_gray import binary_to_float, binary_to_gray, format_to_n_bits, \
  mutate_n_bits_from_binary_or_gray, range_of_numbers_binary_and_gray, \
  generate_random_binary_with_a_len, mutation_binary_or_gray_by_flipping,\
  get_float_from_custom_representation, get_binary_from_custom_representation, \
  get_gray_from_custom_representation
from evo_tools.helpers import sub_strings_by_array
from evo_tools.models import Individual, SubPopulation
from evo_tools.selection import select_parents_by_fitness_proportionate, \
  select_parents_by_roulette, select_parents_by_tournament

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

    variables_array = self._variables.split()

    if (len(variables_array) != len(self._sub_populations)):
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
          Individual(
            binary,
            gray,
            0,
            bits,
            numbers,
            sympify(str(self._function)),
            self._variables.split()
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
    self._sort_population_by_score(self._current_population)

  def _select(
    self,
    individuals: List[Individual],
    minimize: bool,
    sample_size: int,
    mutation_method: str
  ) -> None:
    mutated_individuals = self._mutation(individuals, mutation_method)
    self._fitness(mutated_individuals, minimize)
    self._sort_population_by_score(mutated_individuals)

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

  def _validate_binaries_in_range(self, binaries: List[List[str]]) -> bool:
    """
    Method that validates if a given list of binaries are in the domain.

    Args:
      binaries: List[List[str]]
        List genotypes from each SubPopulation.

    Returns:
      bool: whether or not binaries are valid
    """
    for binary in binaries:
      for i, gen in enumerate(binary):
        try:
          binary_to_float(
            gen,
            self._sub_populations[i].numbers_dict,
            self._sub_populations[i].rng,
            self._precision
          )
        except:
          return False

    return True

  def _parents_selection_by_fitness_proportionate(
    self,
    seed: float
  ) -> List[Tuple[Individual, Individual]]:
    return select_parents_by_fitness_proportionate(self._current_population, seed)

  def _parents_selection_by_roulette(
    self,
    seed: float
  ):
    return select_parents_by_roulette(self._current_population, seed)

  def _parents_selection_by_tournament(
    self,
    seed: float,
    K: int,
    minimize: bool
  ):
    return select_parents_by_tournament(self._current_population, seed, K, minimize)

  def _crossover_one_point(
    self,
    seed: float,
    parent_selection_method,
    minimize: bool
  ) -> List[Individual]:
    """
    Method that creates n children from 2 * n parents combining their genotype
    based in the crossover probability.

    Raises:
      Exception: when the initial data wasn't selected

    Returns:
      List[Individual]: n children
    """
    parents = self._generate_parents_using_a_method(
      seed,
      parent_selection_method,
      minimize
    )

    if len(parents) == 0:
      return []

    total_bits = parents[0][0].get_total_bits()
    bits = parents[0][0].get_bits()
    children: List[Individual] = []

    for p_parent in parents:
      if random() < self._crossover_rate:
        points = [i for i in range(0, total_bits)]
        p1, p2 = p_parent
        binary_p1, binary_p2 = p1.get_binary(), p2.get_binary()
        gray_p1, gray_p2 = p1.get_gray(), p2.get_gray()
        binary_children: List[str] = []
        gray_children: List[str] = []
        attempts = 0
        are_first_binaries_valid = False
        are_second_binaries_valid = False

        while True:
          point = choice(points)
          binary_children = [
            binary_p1[:point] + binary_p2[point:],
            binary_p2[:point] + binary_p1[point:]
          ]
          gray_children = [
            gray_p1[:point] + gray_p2[point:],
            gray_p2[:point] + gray_p1[point:]
          ]
          first_binaries_to_validate = [
            sub_strings_by_array(binary_children[0], bits)
          ]
          second_binaries_to_validate = [
            sub_strings_by_array(binary_children[1], bits)
          ]
          are_first_binaries_valid = self._validate_binaries_in_range(
            first_binaries_to_validate
          )
          are_second_binaries_valid = self._validate_binaries_in_range(
            second_binaries_to_validate
          )

          if are_first_binaries_valid or are_second_binaries_valid or attempts >= total_bits:
            break
          else:
            points = [p for p in points if p != point]

            if len(points) == 0:
              break

          attempts += 1

        if are_first_binaries_valid:
          children.append(
            Individual(
              binary_children[0],
              gray_children[0],
              0,
              bits,
              self._get_fen(binary_children[0], bits), # type: ignore
              sympify(str(self._function)),
              self._variables.split()
            )
          )

        if are_second_binaries_valid:
          children.append(
            Individual(
              binary_children[1],
              gray_children[1],
              0,
              bits,
              self._get_fen(binary_children[1], bits), # type: ignore
              sympify(str(self._function)),
              self._variables.split()
            )
          )

    return children

  def _crossover_two_points(
    self,
    seed: float,
    parent_selection_method,
    minimize: bool
  ) -> List[Individual]:
    parents = self._generate_parents_using_a_method(
      seed,
      parent_selection_method,
      minimize
    )

    if len(parents) == 0:
      return []

    total_bits = parents[0][0].get_total_bits()
    bits = parents[0][0].get_bits()
    children: List[Individual] = []

    for p_parent in parents:
      if random() < self._crossover_rate:
        points = [i for i in range(0, total_bits)]
        p1, p2 = p_parent
        binary_p1, binary_p2 = p1.get_binary(), p2.get_binary()
        gray_p1, gray_p2 = p1.get_gray(), p2.get_gray()
        binary_children: List[str] = []
        gray_children: List[str] = []
        attempts = 0
        are_first_binaries_valid = False
        are_second_binaries_valid = False

        while True:
          p1, p2 = sample(points, 2)
          p_max, p_min = p1, p2

          if p_max < p2:
            p_max = p2

          if p_min > p1:
            p_min = p1

          binary_children = [
            binary_p1[:p_min] + binary_p2[p_min:p_max] + binary_p1[p_max:],
            binary_p2[:p_min] + binary_p1[p_min:p_max] + binary_p2[p_max:]
          ]
          gray_children = [
            gray_p1[:p_min] + gray_p2[p_min:p_max] + gray_p1[p_max:],
            gray_p2[:p_min] + gray_p1[p_min:p_max] + gray_p2[p_max:]
          ]
          first_binaries_to_validate = [
            sub_strings_by_array(binary_children[0], bits)
          ]
          second_binaries_to_validate = [
            sub_strings_by_array(binary_children[1], bits)
          ]
          are_first_binaries_valid = self._validate_binaries_in_range(
            first_binaries_to_validate
          )
          are_second_binaries_valid = self._validate_binaries_in_range(
            second_binaries_to_validate
          )

          if are_first_binaries_valid or are_second_binaries_valid or attempts >= total_bits:
            break

          attempts += 1

        if are_first_binaries_valid:
          children.append(
            Individual(
              binary_children[0],
              gray_children[0],
              0,
              bits,
              self._get_fen(binary_children[0], bits), # type: ignore
              sympify(str(self._function)),
              self._variables.split()
            )
          )

        if are_second_binaries_valid:
          children.append(
            Individual(
              binary_children[1],
              gray_children[1],
              0,
              bits,
              self._get_fen(binary_children[1], bits), # type: ignore
              sympify(str(self._function)),
              self._variables.split()
            )
          )

    return children

  def _crossover_uniform(
    self,
    seed: float,
    parent_selection_method,
    minimize: bool
  ):
    parents = self._generate_parents_using_a_method(
      seed,
      parent_selection_method,
      minimize
    )

    if len(parents) == 0:
      return []

    total_bits = parents[0][0].get_total_bits()
    bits = parents[0][0].get_bits()
    children: List[Individual] = []

    for p_parent in parents:
      if random() < self._crossover_rate:
        p0, p1 = p_parent
        binary_p0, binary_p1 = p0.get_binary(), p1.get_binary()
        gray_p0, gray_p1 = p0.get_gray(), p1.get_gray()
        mask = generate_random_binary_with_a_len(total_bits)
        binary_children: List[str] = ['', '']
        gray_children: List[str] = ['', '']

        for i, mask_element in enumerate(mask):
          if mask_element == '0':
            binary_children[0] += binary_p0[i]
            gray_children[0] += gray_p0[i]

            binary_children[1] += binary_p1[i]
            gray_children[1] += gray_p1[i]
          else:
            binary_children[0] += binary_p1[i]
            gray_children[0] += gray_p1[i]

            binary_children[1] += binary_p0[i]
            gray_children[1] += gray_p0[i]

        first_binaries_to_validate = [
          sub_strings_by_array(binary_children[0], bits)
        ]
        second_binaries_to_validate = [
          sub_strings_by_array(binary_children[1], bits)
        ]
        are_first_binaries_valid = self._validate_binaries_in_range(
          first_binaries_to_validate
        )
        are_second_binaries_valid = self._validate_binaries_in_range(
          second_binaries_to_validate
        )

        if are_first_binaries_valid:
          children.append(
            Individual(
              binary_children[0],
              gray_children[0],
              0,
              bits,
              self._get_fen(binary_children[0], bits), # type: ignore
              sympify(str(self._function)),
              self._variables.split()
            )
          )

        if are_second_binaries_valid:
          children.append(
            Individual(
              binary_children[1],
              gray_children[1],
              0,
              bits,
              self._get_fen(binary_children[1], bits), # type: ignore
              sympify(str(self._function)),
              self._variables.split()
            )
          )

    return children

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
      mutated_children.append(child)
      attempts = 0

      if random() < self._mutation_rate:
        if (self._print):
          print(f'  Mutation for child: {child}\n')

        while True:
          binary = self._do_mutation_using_a_method(
            mutation_method,
            child.get_binary()
          )
          bits = child.get_bits()
          gray = format_to_n_bits(
            binary_to_gray(binary),
            sum(bits)
          )
          binaries_to_validate = [
            sub_strings_by_array(binary, bits),
            sub_strings_by_array(gray, bits)
          ]
          are_binaries_valid = self._validate_binaries_in_range(
            binaries_to_validate
          )

          if are_binaries_valid:
            mutated_child = Individual(
              binary,
              gray,
              0,
              bits,
              self._get_fen(binary, bits), # type: ignore
              sympify(str(self._function)),
              self._variables.split()
            )
            mutated_children.pop()
            mutated_children.append(mutated_child)

            if (self._print):
              print(f'  Mutation for child completed: {mutated_child}\n')

            break
          elif attempts > 5:
            break

          attempts += 1

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
    variables_array = self._variables.split()

    if len(population_sample) == 0:
      return

    bits = population_sample[0].get_bits()
    function_evaluations: List[float] = []

    for i, individual in enumerate(population_sample):
      individual.set_score(0)
      individual.set_objective_value(None)
      chromosome = individual.get_binary()

      if (self._print):
        print(f'Chromosome {i}: {chromosome}')

      gens = sub_strings_by_array(chromosome, bits)
      fens: List[float] = []

      for i, gen in enumerate(gens):
        try:
          fen = float(
            binary_to_float(
              gen,
              self._sub_populations[i].numbers_dict,
              self._sub_populations[i].rng,
              self._precision
            )
          )
          fens.append(fen)
        except:
          pass

      if (self._print):
        print(f'  gens: {gens}')
        print(f'  fens: {fens}')

      function: exp = sympify(str(self._function))

      if len(gens) == len(fens):
        # Evaluate the given function variable per variable
        for i, v in enumerate(variables_array):
          function = function.subs(v, fens[i])  # type: ignore

        objective_value = float(function)
        function_evaluations.append(objective_value)
        individual.set_objective_value(objective_value)

        if self._print:
          print(f'  fitness: {objective_value}\n')
      elif self._print:
        print(f'  fitness: Fail\n')

    valid_population_sample = [
      individual for individual in population_sample if individual._objective_value is not None
    ]

    if len(function_evaluations) == 0:
      return

    reference_value = max(function_evaluations) if minimize else min(function_evaluations)

    for i, objective_value in enumerate(function_evaluations):
      score = (
        1e-3 + reference_value - objective_value
        if minimize else
        1e-3 + objective_value - reference_value
      )
      valid_population_sample[i].set_score(score)

  def _sort_population_by_score(self, population_sample: List[Individual]) -> None:
    population_sample.sort(reverse = True, key = lambda x: x.get_score())

  def _generate_parents_using_a_method(
    self,
    seed: float,
    parent_selection_method,
    minimize: bool
  ):
    if parent_selection_method == 'fitness_proportionate':
      return self._parents_selection_by_fitness_proportionate(seed)

    if parent_selection_method == 'roulette':
      return self._parents_selection_by_roulette(seed)

    if parent_selection_method == 'tournament':
      return self._parents_selection_by_tournament(seed, 10, minimize)

    raise Exception('Parent selection method not allowed')

  def _validate_parents_selection_methods(
    self,
    parent_selection_method
  ):
    if parent_selection_method in ['fitness_proportionate', 'roulette', 'tournament']:
      return

    raise Exception('Method not allowed')

  def _do_crossover_using_a_method(
    self,
    seed: float,
    crossover_method,
    parent_selection_method,
    minimize: bool
  ):
    if crossover_method == 'one_point':
      return self._crossover_one_point(seed, parent_selection_method, minimize)

    if crossover_method == 'two_points':
      return self._crossover_two_points(seed, parent_selection_method, minimize)

    if crossover_method ==  'uniform':
      return self._crossover_uniform(seed, parent_selection_method, minimize)

    raise Exception('Crossover method not allowed')

  def _validate_crossover_methods(
    self,
    crossover_method
  ):
    if crossover_method in ['one_point', 'uniform', 'two_points']:
      return

    raise Exception('Crossover method not allowed')

  def _do_mutation_using_a_method(
    self,
    mutation_method,
    binary_or_gray: str
  ):
    if mutation_method == 'one_point':
      return mutate_n_bits_from_binary_or_gray(binary_or_gray)

    if mutation_method == 'two_points':
      return mutate_n_bits_from_binary_or_gray(binary_or_gray, 2)

    if mutation_method ==  'flipping':
        return mutation_binary_or_gray_by_flipping(binary_or_gray)

    raise Exception('Mutation method not allowed')

  def _validate_mutation_methods(
    self,
    mutation_method
  ):
    if mutation_method in ['one_point', 'two_points', 'flipping']:
      return

    raise Exception('Mutation method not allowed')

  def _get_fen(self, binary_or_gray: str, bits: List[int]):
    binaries = sub_strings_by_array(
      binary_or_gray,
      bits
    )
    numbers = '['

    for i, binary in enumerate(binaries):
      try:
        fen = binary_to_float(
          binary,
          self._sub_populations[i].numbers_dict,
          self._sub_populations[i].rng,
          self._precision
        )
        numbers += f'{fen}, '
      except:
        pass

    strToReplace = ', '
    replacementStr = ']'
    strToReplaceReversed = strToReplace[::-1]
    replacementStrReversed = replacementStr[::-1]
    numbers = numbers[::-1].replace(strToReplaceReversed, replacementStrReversed, 1)[::-1]

    return numbers

  def canonical_algorithm(
    self,
    ITERATIONS = 100,
    MINIMIZE = True,
    SEED = 1.1,
    PRINT = False,
    PARENT_SELECTION_METHOD = 'fitness_proportionate',
    CROSSOVER_METHOD = 'one_point',
    MUTATION_METHOD = 'one_point'
  ) -> Tuple[List[float], Dict[str, str], exp, List[float]]:
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
      Tuple[List[float], Dict[str, str], exp, fitness_avg_list]: A tuple that
      contains the list of historical scores from each generation, a Dict with
      the solution for each given variable, the result calculated for the
      obtained solution and the average of the fitness per each generation.
    """
    self._validate_parents_selection_methods(PARENT_SELECTION_METHOD)
    self._validate_crossover_methods(CROSSOVER_METHOD)
    self._validate_mutation_methods(MUTATION_METHOD)

    self._select_initial_population()
    start = time()
    self._fitness(self._current_population, MINIMIZE)
    self._sort_population_by_score(self._current_population)
    self._best_individual = self._current_population[0]
    current_iteration = 1
    scores: List[float] = []
    fitness_avg_list: List[float] = [
      np.mean(
        np.array(
          list(
            map(
              lambda individual: individual.get_fitness(),
              self._current_population
            )
          ) # type: ignore
        )
      )
    ]
    end = time()

    if PRINT:
      print(
        f'\n{current_iteration}º iteration.\nBest individual: {self._best_individual}.\nSelection strength: {self._selection_strength}.\nTime elapsed: {end - start}s.'
      )
      df = pd.DataFrame(loads(str(self._current_population)))
      print(df, end = '\n\n')

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
      fitness_avg_list.append(
        np.mean(
          np.array(
            list(
              map(
                lambda individual: individual.get_fitness(),
                self._current_population
              )
            ) # type: ignore
          )
        )
      )
      end = time()

      if self._selection_strength <= 1e-4:
        break

      if PRINT:
        print(
          f'\n{current_iteration}º iteration.\nBest individual: {self._best_individual}.\nSelection strength: {self._selection_strength}.\nTime elapsed: {end - start}s.'
        )
        df = pd.DataFrame(loads(str(self._current_population)))
        print(df, end = '\n\n')

    if PRINT:
      print(
        f'\n\nFinally:\n{current_iteration}º iteration.\nBest individual: {self._best_individual}.\nSelection strength: {self._selection_strength}.'
      )

    binaries = sub_strings_by_array(
      self._best_individual.get_binary(),
      self._best_individual.get_bits()
    )
    floats: List[str] = []

    for i, binary in enumerate(binaries):
      try:
        fen = binary_to_float(
          binary,
          self._sub_populations[i].numbers_dict,
          self._sub_populations[i].rng,
          self._precision
        )
        floats.append(fen)
      except:
        pass

    if len(floats) == 0:
      raise Exception('Something went wrong')

    variables_array = self._variables.split()
    function = sympify(str(self._function))
    solution: Dict[str, str] = {}

    for i, v in enumerate(variables_array):
      function = function.subs(v, floats[i])
      solution[v] = floats[i]

    if PRINT:
      print('Solution:')
      print(f'  Variables: {solution}')
      print(f'  Evaluation: {function}')
      print(f'  Fitness average: {fitness_avg_list}')

    return scores, solution, function, fitness_avg_list
