import numpy as np
import pandas as pd
from json import loads
from random import choice, sample, random
from math import log
from functools import reduce
from sympy import exp, sympify
from typing import Dict, List, Tuple, Union, Literal

from evo_tools.bin_gray import NumberBinaryGrayRepresentation, binary_to_float, binary_to_gray, format_to_n_bits, mutate_n_bits_from_binary_or_gray, range_of_numbers_binary_and_gray, generate_random_binary_with_a_len, mutation_binary_or_gray_by_flipping
from evo_tools.helpers import sub_strings_by_array

ParentSelectionMethods = Literal['fitness_proportionate', 'roulette', 'tournament']
CrossoverMethods = Literal['one_point', 'two_points', 'uniform']
MutationMethods = Literal['one_point', 'two_points', 'flipping']

class Individual():
  """
  A member of a population.
  """
  def __init__(
    self,
    binary: str,
    gray: str,
    score: float,
    bits: List[int],
    numbers: str,
    function,
    variables_array: List[str]
  ) -> None:
    self._binary = binary
    self._gray = gray
    self._score = score
    self._bits = bits
    self._numbers = numbers
    self._function = function
    self._variables_array = variables_array

  def get_binary(self) -> str:
    return self._binary

  def get_gray(self) -> str:
    return self._gray

  def get_score(self) -> float:
    return float(self._score)

  def get_bits(self) -> List[int]:
    return self._bits.copy()

  def get_total_bits(self) -> int:
    return reduce(lambda a, b: a + b, self._bits)

  def set_score(self, score: float) -> None:
    self._score = score

  def get_numbers(self) -> str:
    return self._numbers

  def get_fitness(self):
    numbers = loads(self._numbers)
    f = self._function.copy()

    for i, n in enumerate(numbers):
      f = f.subs(self._variables_array[i], n)

    return f

  def _str_bits(self, bits: List[int]) -> str:
    result = '['

    for i, bit in enumerate(bits):
      if i != len(bits) - 1:
        result += f'{bit}, '
      else:
        result += f'{bit}]'

    return result

  def __str__(self) -> str:
    return f'{{ \
"binary": "{self._binary}", \
"gray": "{self._gray}", \
"numbers": "{self._numbers}", \
"bits": "{self._str_bits(self._bits)}", \
"score": "{self._score}", \
"fitness": "{self.get_fitness()}" \
}}'

  def __repr__(self) -> str:
    return str(self)

class SubPopulation():
  """
  A class to represent a SubPopulation
  --

  A SubPopulation is nothing but a object that represents a real range (float interval).
  So, a Population is build with several ranges, with its representation in binary
  and gray code and the number of bits that are used to represent the range.

  For example, lets say you want to create a Population of one range: [1, 2],
  with a precision of 0.1, then we will only have an array of SubPopulation,
  whose len will be one, and that only member will store its class attributes as follows:

  Attributes
  --

  rng: Tuple[Union[float, int], Union[float, int]]
    The range specified for this SubPopulation, for this case (1, 2)

  numbers: List[NumberBinaryGrayRepresentation]

  bits: int
    Number of bits used for represent the float value.
  """
  def __init__(
    self,
    rng: Tuple[Union[float, int], Union[float, int]],
    numbers: List[NumberBinaryGrayRepresentation],
    bits: int,
  ) -> None:
    self.rng = rng
    self.numbers = numbers
    self.bits = bits

  def __str__(self) -> str:
    return f'{{ "rng": {self.rng}, "numbers": {self.numbers}, "bits": {self.bits} }}'

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

    p10 = 1 if precision == 1 else pow(precision, -1)
    self._n_decimal_digits = int(round(log(p10, 10)))

    for rng in ranges:
      sub_population_range, bits = range_of_numbers_binary_and_gray(
        rng,
        self._precision
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

  def print(self):
    """
    Prints the current Population data. The current sample and the data from each
    SubPopulation.
    """
    print('\nCurrent population sample:\n')
    print(self._current_population)
    print()
    print('\nData from population members:')

    for i, sub_population in enumerate(self._sub_populations):
      print(f'  {i + 1}. Range:', sub_population.rng)
      print(f'  {i + 1}. Bits:', sub_population.bits)
      print(f'  {i + 1}. Numbers:', sub_population.numbers)
      print()

  def _select_initial_population(self, sample_size: int) -> List[Individual]:
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
    self._sample_size = sample_size

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
      samples: List[Tuple[List[NumberBinaryGrayRepresentation], int]] = []

      for sub_population in self._sub_populations:
        samples.append((
          sample(sub_population.numbers, sample_size),
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
          binary += current_sample[i].get_binary()
          gray += current_sample[i].get_gray()

          if j != len(samples) - 1:
            numbers += f'{current_sample[i].get_number()}, '
          else:
            numbers += f'{current_sample[i].get_number()}]'

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

  def _update_current_population(self, new_population: List[Individual]) -> None:
    """
    Method that updates the current sample, after crossover or mutation.

    Args:
      new_population (List[Individual])
    """
    if (len(new_population) > self._sample_size):
      self._current_population = new_population[:self._sample_size]

      return

    self._current_population = new_population

  def _select(
    self,
    individuals: List[Individual],
    minimize: bool,
    sample_size: int
  ) -> None:
    mutated_individuals = self._mutation(individuals)
    self._fitness(mutated_individuals, minimize)
    mutated_individuals.sort(reverse = minimize, key = lambda x: x.get_score())

    # Calculate the mean and std of the population before the selection
    current_population_score = np.array([
      x.get_score() for x in self._current_population
    ])
    score_std_before_selection = np.std(current_population_score)
    score_mean_before_selection = np.mean(current_population_score)

    self._update_current_population(
      self._current_population[
        :len(self._current_population) - len(mutated_individuals)
      ] + mutated_individuals[:sample_size]
    )

    # Calculate the mean of the population after the selection
    current_population_score = np.array([
      x.get_score() for x in self._current_population
    ])
    score_mean_after_selection = np.mean(current_population_score)
    self._selection_strength = abs(
      (score_mean_after_selection - score_mean_before_selection) / score_std_before_selection
    ) if score_std_before_selection > 0 else 0

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
          _range = self._sub_populations[i].rng
          fen = binary_to_float(gen, _range, self._precision)
          x0, xf = _range

          if float(fen.get_number()) < x0 or float(fen.get_number()) > xf:
            return False
        except:
          return False

    return True

  def _parents_selection_by_fitness_proportionate(
    self,
    seed: float
  ) -> List[Tuple[Individual, Individual]]:
    """
    Method that selects parents based on the following probability:
      (own_score + K)/(generation_score + K*len(scores)).
    Both parents have to be different from each other at least in one bit.

    Args:
      seed (float): value that will be rounded to int. This value will set the
      maximum amount of parents to be generated.

    Returns:
      List[Tuple[Individual, Individual]]: list that contains a pair of
      different parents.
    """
    total_population = len(self._current_population)
    parents_candidates = np.array(self._current_population)
    generation_scores = np.array(
      [individual.get_score() for individual in parents_candidates]
    )
    generation_score = sum(generation_scores)
    random_parents_indexes_chosen = np.random.choice(
      total_population,
      size = (round(seed), 2),
      p = np.array(
        [x + generation_score / 2 for x in generation_scores]
      ) / (generation_score + generation_score / 2 * len(generation_scores))
    )
    unique_random_parents_indexes_chosen, _ = np.unique(
      [
        str(
          np.ndarray.tolist(index)
        )[1:-1].replace(' ', '') for index in random_parents_indexes_chosen
      ],
      return_index = True
    )
    final_parents_indexes = list(
      filter(
        lambda a: a[0] != a[1],
        map(
          lambda e: [int(i) for i in e.split(',')],
          unique_random_parents_indexes_chosen
        )
      )
    )
    parents: List[Tuple[Individual, Individual]] = []

    for indexes in final_parents_indexes:
      i_1, i_2 = indexes
      parents.append((
        self._current_population[i_1],
        self._current_population[i_2]
      ))

    return parents

  def _parents_selection_by_roulette(
    self,
    seed: float
  ):
    """
    Method that selects parents based in the following probability:
      own_score/generation_score.
    Both parents have to be different from each other at least in one bit.

    Args:
      seed (float): value that will be rounded to int. This value will set the
      maximum amount of parents to be generated.

    Returns:
      List[Tuple[Individual, Individual]]: list that contains a pair of
      different parents.
    """
    total_population = len(self._current_population)
    parents_candidates = np.array(self._current_population)
    generation_scores = np.array(
      [individual.get_score() for individual in parents_candidates]
    )
    generation_score = sum(generation_scores)
    random_parents_indexes_chosen = np.random.choice(
      total_population,
      size = (round(seed), 2),
      p = generation_scores / generation_score
    )
    unique_random_parents_indexes_chosen, _ = np.unique(
      [
        str(
          np.ndarray.tolist(index)
        )[1:-1].replace(' ', '') for index in random_parents_indexes_chosen
      ],
      return_index = True
    )
    final_parents_indexes = list(
      filter(
        lambda a: a[0] != a[1],
        map(
          lambda e: [int(i) for i in e.split(',')],
          unique_random_parents_indexes_chosen
        )
      )
    )
    parents: List[Tuple[Individual, Individual]] = []

    for indexes in final_parents_indexes:
      i_1, i_2 = indexes
      parents.append((
        self._current_population[i_1],
        self._current_population[i_2]
      ))

    return parents

  def _parents_selection_by_tournament(
    self,
    seed: float,
    K: int
  ):
    parents: List[Tuple[Individual, Individual]] = []

    while len(parents) < seed:
      chosen_list: List[Individual] = []

      for _ in range(2):
        candidates = sample(self._current_population, K)
        chosen = min(
          candidates,
          key = lambda individual: abs(individual.get_fitness())
        )
        chosen_list.append(chosen)

      parents.append(tuple(chosen_list))

    return parents

  def _crossover_one_point(
    self,
    seed: float,
    parent_selection_method: ParentSelectionMethods
  ) -> List[Individual]:
    """
    Method that creates n children from 2 * n parents combining their genotype
    based in the crossover probability.

    Raises:
      Exception: when the initial data wasn't selected

    Returns:
      List[Individual]: n children
    """
    parents = self._generate_parents_using_a_method(seed, parent_selection_method)

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
            sub_strings_by_array(binary_children[0], bits),
            sub_strings_by_array(gray_children[0], bits),
          ]
          second_binaries_to_validate = [
            sub_strings_by_array(binary_children[1], bits),
            sub_strings_by_array(gray_children[1], bits)
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
            points = list(filter(lambda number: number != point, points))

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
    parent_selection_method: ParentSelectionMethods
  ) -> List[Individual]:
    parents = self._generate_parents_using_a_method(seed, parent_selection_method)

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
            sub_strings_by_array(binary_children[0], bits),
            sub_strings_by_array(gray_children[0], bits),
          ]
          second_binaries_to_validate = [
            sub_strings_by_array(binary_children[1], bits),
            sub_strings_by_array(gray_children[1], bits)
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
    parent_selection_method: ParentSelectionMethods
  ):
    parents = self._generate_parents_using_a_method(seed, parent_selection_method)

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
          sub_strings_by_array(binary_children[0], bits),
          sub_strings_by_array(gray_children[0], bits),
        ]
        second_binaries_to_validate = [
          sub_strings_by_array(binary_children[1], bits),
          sub_strings_by_array(gray_children[1], bits)
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
    mutation_method: MutationMethods = 'one_point'
  ) -> List[Individual]:
    """
    Method that changes 1 bit from a children based on the mutation probability.

    Args:
      children (List[Individual]): children mutated
    """
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
            reduce(lambda a, b: a + b, bits)
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

  def _fitness(self, population_sample: List[Individual], minimize = True) -> None:
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
    bits = population_sample[0].get_bits()
    function_evaluations = []

    for i, individual in enumerate(population_sample):
      chromosome = individual.get_binary()

      if (self._print):
        print(f'Chromosome {i}: {chromosome}')

      gens = sub_strings_by_array(chromosome, bits)
      fens: List[float] = []

      for i, gen in enumerate(gens):
        _range = self._sub_populations[i].rng

        try:
          fen = float(
            binary_to_float(gen, _range, self._precision).get_number()
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

        function_evaluations.append(function)

        if self._print:
          print(f'  fitness: {function}\n')
      else:
        function_evaluations.append('Fail')

        if self._print:
          print(f'  fitness: Fail\n')

    final_function_evaluations = []
    final_population_sample = []

    for i, fe in enumerate(function_evaluations):
      if fe != 'Fail':
        final_function_evaluations.append(fe)
        final_population_sample.append(population_sample[i])

    population_sample = final_population_sample

    if len(final_function_evaluations) == 0:
      return

    if minimize:
      maxi = max([abs(e) for e in final_function_evaluations])

      for i, e in enumerate(final_function_evaluations):
        score = 1e-3 + maxi - abs(e)
        population_sample[i].set_score(score)
    else:
      mini = min([abs(e) for e in final_function_evaluations])

      for i, e in enumerate(final_function_evaluations):
        score = 1e-3 + abs(e) - mini
        population_sample[i].set_score(score)

  def _generate_parents_using_a_method(
    self,
    seed: float,
    parent_selection_method: ParentSelectionMethods
  ):
    match parent_selection_method:
      case 'fitness_proportionate':
        return self._parents_selection_by_fitness_proportionate(seed)
      case 'roulette':
        return self._parents_selection_by_roulette(seed)
      case 'tournament':
        return self._parents_selection_by_tournament(seed, K = 10)
      case _:
        raise Exception('Parent selection method not allowed')

  def _validate_parents_selection_methods(
    self,
    parent_selection_method: ParentSelectionMethods
  ):
    match parent_selection_method:
      case 'fitness_proportionate':
        return
      case 'roulette':
        return
      case 'tournament':
        return
      case _:
        raise Exception('Method not allowed')

  def _do_crossover_using_a_method(
    self,
    seed: float,
    crossover_method: CrossoverMethods,
    parent_selection_method: ParentSelectionMethods
  ):
    match crossover_method:
      case 'one_point':
        return self._crossover_one_point(seed, parent_selection_method)
      case 'two_points':
        return self._crossover_two_points(seed, parent_selection_method)
      case 'uniform':
        return self._crossover_uniform(seed, parent_selection_method)
      case _:
        raise Exception('Crossover method not allowed')

  def _validate_crossover_methods(
    self,
    crossover_method: CrossoverMethods
  ):
    match crossover_method:
      case 'one_point':
        return
      case 'uniform':
        return
      case 'two_points':
        return
      case _:
        raise Exception('Crossover method not allowed')

  def _do_mutation_using_a_method(
    self,
    mutation_method: MutationMethods,
    binary_or_gray: str
  ):
    match mutation_method:
      case 'one_point':
        return mutate_n_bits_from_binary_or_gray(binary_or_gray)
      case 'two_points':
        return mutate_n_bits_from_binary_or_gray(binary_or_gray, 2)
      case 'flipping':
        return mutation_binary_or_gray_by_flipping(binary_or_gray)
      case _:
        raise Exception('Mutation method not allowed')

  def _validate_mutation_methods(
    self,
    mutation_method: MutationMethods
  ):
    match mutation_method:
      case 'one_point':
        return
      case 'two_points':
        return
      case 'flipping':
        return
      case _:
        raise Exception('Mutation method not allowed')

  def _get_fen(self, binary_or_gray: str, bits: List[int]):
    binaries = sub_strings_by_array(
      binary_or_gray,
      bits
    )
    ranges: List[Tuple[float | int, float | int]] = []

    for i, binary in enumerate(binaries):
      try:
        _range = self._sub_populations[i].rng
        fen = binary_to_float(binary, _range, self._precision)
        x0, xf = _range

        if float(fen.get_number()) < x0 or float(fen.get_number()) > xf:
          pass
        else:
          ranges.append(_range)
      except:
        pass

    numbers = '['

    for i, rng in enumerate(ranges):
      binary = binaries[i]

      try:
        fen = binary_to_float(binary, rng, self._precision).get_number()
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
    SAMPLE_SIZE: int,
    ITERATIONS = 100,
    MINIMIZE = True,
    SEED = 1.5,
    PRINT = False,
    PARENT_SELECTION_METHOD: ParentSelectionMethods = 'fitness_proportionate',
    CROSSOVER_METHOD: CrossoverMethods = 'one_point',
    MUTATION_METHOD: MutationMethods = 'one_point'
  ) -> Tuple[List[float], Dict[str, str], exp]:
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
      Tuple[List[float], Dict[str, str], exp]: A tuple that contains the list of
      historical scores from each generation, a Dict with the solution for each
      given variable and the result calculated for the obtained solution.
    """
    self._validate_parents_selection_methods(PARENT_SELECTION_METHOD)
    self._validate_crossover_methods(CROSSOVER_METHOD)
    self._validate_mutation_methods(MUTATION_METHOD)

    self._select_initial_population(SAMPLE_SIZE)
    self._fitness(self._current_population, MINIMIZE)
    self._current_population.sort(
      reverse = MINIMIZE,
      key = lambda x: x.get_score()
    )
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

    if PRINT:
      print(
        f'{current_iteration}º iteration, best individual: {self._best_individual}, selection strength: {self._selection_strength}.'
      )
      df = pd.DataFrame(loads(str(self._current_population)))
      print(df)

    for i in range(ITERATIONS - 1):
      current_iteration += 1
      children = self._do_crossover_using_a_method(
        SAMPLE_SIZE * SEED,
        CROSSOVER_METHOD,
        PARENT_SELECTION_METHOD
      )
      self._select(children, MINIMIZE, SAMPLE_SIZE)
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

      if self._selection_strength <= 1e-4:
        break

      if round(self._best_individual.get_score(), 3) <= 1e-4:
        break

      if PRINT:
        print(
          f'{current_iteration}º iteration, best individual: {self._best_individual}, selection strength: {self._selection_strength}.'
        )
        df = pd.DataFrame(loads(str(self._current_population)))
        print(df)

    if PRINT:
      print(
        f'\n\nFinally:\n{current_iteration}º iteration, best individual: {self._best_individual}, selection strength: {self._selection_strength}.'
      )

    binaries = sub_strings_by_array(
      self._best_individual.get_binary(),
      self._best_individual.get_bits()
    )
    ranges: List[Tuple[float | int, float | int]] = []

    for i, binary in enumerate(binaries):
      try:
        _range = self._sub_populations[i].rng
        fen = binary_to_float(binary, _range, self._precision)
        x0, xf = _range

        if float(fen.get_number()) < x0 or float(fen.get_number()) > xf:
          pass
        else:
          ranges.append(_range)
      except:
        pass

    if len(ranges) == 0:
      raise Exception('Something went wrong')

    variables_array = self._variables.split()
    floats: List[str] = []

    for i, rng in enumerate(ranges):
      binary = binaries[i]
      fen = binary_to_float(binary, rng, self._precision)
      floats.append(fen.get_number())

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

    return scores, solution, function
