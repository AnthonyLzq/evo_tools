import numpy as np
from random import choice, sample, random
from math import log
from functools import reduce
from sympy import exp, sympify
from typing import Dict, List, Tuple, Union

from evo_tools.bin_gray import NumberBinaryGrayRepresentation, binary_to_float, binary_to_gray, format_to_n_bits, mutate_binary_or_gray, range_of_numbers_binary_and_gray
from evo_tools.helpers import sub_strings_by_array

class Individual():
  """
  A member of a population.
  """
  def __init__(
    self,
    binary: str,
    gray: str,
    score: float,
    bits: List[int]
  ) -> None:
    self._binary = binary
    self._gray = gray
    self._score = score
    self._bits = bits

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

  def __str__(self) -> str:
    return f'{{ "binary": "{self._binary}", "gray": "{self._gray}", "score": "{self._score}", "bits": {self._bits} }}'

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
      binaries = []
      grays = []
      scores = []

      for sub_population in self._sub_populations:
        samples.append((
          sample(sub_population.numbers, sample_size),
          sub_population.bits
        ))

      first_sample, _ = samples[0]

      for i, __ in enumerate(first_sample):
        binary: str = ''
        gray: str = ''
        bits: List[int] = []

        for s in samples:
          current_sample, current_bits = s
          bits.append(current_bits)
          binary += current_sample[i].get_binary()
          gray += current_sample[i].get_gray()

        binaries.append(binary)
        grays.append(gray)
        scores.append(0)
        self._initial_population.append(Individual(binary, gray, 0, bits))

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
      new_population: List[Individual]
    """
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
    self._update_current_population(
      self._current_population[
        :len(self._current_population) - len(mutated_individuals)
      ] + mutated_individuals[:sample_size]
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
          _range = self._sub_populations[i].rng
          fen = binary_to_float(gen, _range, self._precision)
          x0, xf = _range

          if float(fen.get_number()) < x0 or float(fen.get_number()) > xf:
            return False
        except:
          return False

    return True

  def _parents_selection_by_fitness_proportionate_selection(
    self,
    seed: float
  ) -> List[Tuple[Individual, Individual]]:
    """
    Method that selects parents based on the probability: own_score/generation_score.
    Both parents have to be different from each other at least in one bit.

    Args:
      seed (float): a float value that will be rounded to int. This value will
      set the maximum amount of parents to be generated.

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

  def _crossover_one_point(self, seed: float) -> List[Individual]:
    """
    Method that creates n children from 2 * n parents combining their genotype
    based in the crossover probability.

    Raises:
      Exception: when the initial data wasn't selected

    Returns:
      List[Individual]: n children
    """
    parents = self._parents_selection_by_fitness_proportionate_selection(seed)

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
            Individual(binary_children[0], gray_children[0], 0, bits)
          )

        if are_second_binaries_valid:
          children.append(
            Individual(binary_children[1], gray_children[1], 0, bits)
          )

    return children

  def _mutation(self, children: List[Individual]) -> List[Individual]:
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
          binary = mutate_binary_or_gray(child.get_binary())
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
            mutated_child = Individual(binary, gray, 0, bits)
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
      # 10010110101, [2, 3, 3, 3]
      chromosome = individual.get_binary()

      if (self._print):
        print(f'Chromosome {i}: {chromosome}')

      gens = sub_strings_by_array(chromosome, bits)
      # ['10', '010', '110', '101']
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

  def canonical_algorithm(
    self,
    SAMPLE_SIZE: int,
    ITERATIONS = 100,
    MINIMIZE = True,
    SEED = 1.5,
    PRINT = False
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

    Raises:
      Exception: when a generation has a individual that is outside from all the
      given intervals.

    Returns:
      Tuple[List[float], Dict[str, str], exp]: A tuple that contains the list of
      historical scores from each generation, a Dict with the solution for each
      given variable and the result calculated for the obtained solution.
    """
    self._select_initial_population(SAMPLE_SIZE)
    self._fitness(self._current_population, MINIMIZE)
    self._current_population.sort(
      reverse = MINIMIZE,
      key = lambda x: x.get_score()
    )
    self._best_individual = self._current_population[0]
    current_iteration = 1
    scores: List[float] = []

    if PRINT:
      print(
        f'{current_iteration}º iteration, best individual: {self._best_individual}'
      )

    for i in range(ITERATIONS - 1):
      current_iteration += 1
      children = self._crossover_one_point(SAMPLE_SIZE * SEED)
      self._select(children, MINIMIZE, SAMPLE_SIZE)
      scores.append(self._best_individual.get_score())

      if round(self._best_individual.get_score(), 3) <= 1e-3:
        break

      if PRINT:
        print(
          f'{current_iteration}º iteration, best individual: {self._best_individual}'
        )

    if PRINT:
      print(
        f'\n\nFinally:\n{current_iteration}º iteration, best individual: {self._best_individual}'
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

    for rng in ranges:
      for binary in binaries:
        try:
          fen = binary_to_float(binary, rng, self._precision)
          floats.append(fen.get_number())
        except:
          floats.append('fail')
          pass

      if len(floats) > 0:
        break

    function: exp = sympify(str(self._function))
    solution: Dict[str, str] = {}

    for i, v in enumerate(variables_array):
      function = function.subs(v, floats[i])  # type: ignore
      solution[v] = floats[i]

    if PRINT:
      print('Solution:')
      print(f'  Variables: {solution}')
      print(f'  Evaluation: {function}')

    return scores, solution, function
