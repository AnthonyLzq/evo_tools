from random import choice, sample, random, randint, shuffle
from math import log
from functools import reduce
from sympy import exp, sympify
from typing import List, Tuple, Union
from sys import version_info
from json import dumps

if version_info >= (3, 8):
  from typing import TypedDict
else:
  from typing_extensions import TypedDict

from evo_tools.bin_gray import NumberBinaryAndGray, binary_to_float, binary_to_gray, format_to_n_bits, mutate_binary_or_gray, range_of_numbers_binary_and_gray
from evo_tools.helpers import sub_strings_by_array, subsets_of_pairs

class Individual():
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
    return self._score

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

  numbers: List[NumberBinaryAndGray]
    Where NumberBinaryAndGray is a dictionary with 3 keys:

      number: str
        Float number with fixed precision based in the precision input. In this
        case 0.1

      binary: str
        The binary representation of one number from to the scaled interval.
        This means that for the interval (1, 2) with precision 0.1, 1 will be
        consider as 0 in binary, 1.1 as 1 in binary and so on.

      gray: str
        Analogous to "binaries", but it is the representation of one number from
        the scaled interval in gray code.

  bits: int
    Number of bits used for represent the float value.
  """
  def __init__(
    self,
    rng: Tuple[Union[float, int], Union[float, int]],
    numbers: List[NumberBinaryAndGray],
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

  _print: bool = False
    Whether or not should print the output in the methods.
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

    p10 = pow(precision, -1) if precision != 1 else 1
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
      samples: List[Tuple[List[NumberBinaryAndGray], int]] = []
      binaries = []
      grays = []
      scores = []

      for sub_population in self._sub_populations:
        samples.append((
          sample(sub_population.numbers, sample_size),
          sub_population.bits
        ))

      f_sample, _ = samples[0]

      for i, __ in enumerate(f_sample):
        binary: str = ''
        gray: str = ''
        bits: List[int] = []

        for j, _ in enumerate(self._sub_populations):
          current_sample, current_bits = samples[j]
          bits.append(current_bits)
          binary += current_sample[i]['binary']
          gray += current_sample[i]['gray']

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

  def _get_sample_from_population(self, sample_size: int) -> List[Individual]:
    """
    Method that selects a new sample randomly base on the following probability:
    fitness(i) / fitness_population

    Args:
      sample_size: int

    Raises:
      Exception: if the fitness was not calculated

    Returns:
      List[Individual]: new current population
    """
    current_population_score = reduce(
      lambda acc, individual: acc + individual.get_score(),
      self._current_population,
      0
    )

    if current_population_score == 0:
      raise Exception('Fitness has to be calculated first.')

    new_population: List[Individual] = []

    for individual in self._current_population:
      if random() < individual.get_score() / current_population_score:
        new_population.append(individual)

    new_population_length = len(new_population)

    if new_population_length < sample_size:
      elements_to_add = sample_size - new_population_length

      for _ in range(elements_to_add):
        index = randint(0, len(self._current_population))
        new_population.append(self._current_population[index])
    elif new_population_length > sample_size:
      elements_to_eliminate = new_population_length - sample_size

      for _ in range(elements_to_eliminate):
        index = randint(0, len(new_population))
        new_population = new_population[:index] + new_population[index + 1:]

    return new_population

  def _update_current_population(self, new_population: List[Individual]) -> None:
    """
    Method that updates the current sample, after crossover or mutation.

    Args:
      new_population: List[Individual]
    """
    self._current_population = new_population

  def _select(self, sample_size: int) -> None:
    """
    Method that creates a new sample and update the current sample with the new one.

    Args:
      sample_size: int

    Raises:
      Exception: when the sample size is too big or the initial data was not selected.
    """
    if (sample_size > self._max_sample_size):
      raise Exception(
        f'Sample size too big, maximum is: {self._max_sample_size}'
      )

    try:
      sample_population = self._get_sample_from_population(sample_size)
      self._update_current_population(sample_population)

      if self._print:
        print('\nSelection: \n')
        print(self._current_population)
    except:
      raise Exception(
        'Select initial data was not invoked at the beginning. It must be.'
      )

  def _validate_binaries_in_range(self, binaries: List[List[str]]) -> bool:
    """
    Method that validates if a given list of binaries are in the domain.

    Args:
      binaries: List[List[str]]
        List genotypes from each SubPopulation.

    Returns:
      bool: whether or not binaries are valid
    """
    for b in binaries:
      for i, gen in enumerate(b):
        try:
          _range = self._sub_populations[i].rng
          fen = binary_to_float(gen, _range, self._precision)
          x0, xf = _range

          if float(fen['number']) < x0 or float(fen['number']) > xf:
            return False
        except:
          return False

    return True

  def _parents_selection_by_roulette(self) -> List[Individual]:
    current_population_score = reduce(
      lambda a, b: a + b.get_score(),
      self._current_population,
      0
    ) + 100 * len(self._current_population)
    parents: List[Individual] = []

    for individual in self._current_population:
      p_i = individual.get_score() + 100 / current_population_score

      if random() < p_i:
        parents.append(individual)

    parents_length = len(parents)

    # If parents length is odd, a random parent will be removed
    if parents_length % 2 != 0:
      index = randint(0, parents_length)
      parents = parents[:index] + parents[index + 1:]
      parents_length -= 1

    return parents

  def _crossover_one_point(self) -> List[Individual]:
    """
    Method that creates n children from 2 * n parents combining their genotype
    based in the crossover probability.

    Raises:
      Exception: when the initial data wasn't selected

    Returns:
      List[Individual]: n children
    """
    parents = self._parents_selection_by_roulette()
    shuffle(parents)
    parents_in_pairs = subsets_of_pairs(parents)
    total_bits = parents[0].get_total_bits()
    bits = parents[0].get_bits()
    children: List[Individual] = []

    for p_parent in parents_in_pairs:
      if random() < self._crossover_rate:
        p1, p2 = p_parent
        binary_p1, binary_p2 = p1.get_binary(), p2.get_binary()
        gray_p1, gray_p2 = p1.get_gray(), p2.get_gray()
        binary_children: List[str] = []
        gray_children: List[str] = []
        attempts = 0

        while True:
          point = randint(0, total_bits - 1)
          binary_children = [
            binary_p1[:point] + binary_p2[point:],
            binary_p2[:point] + binary_p1[point:]
          ]
          gray_children = [
            gray_p1[:point] + gray_p2[point:],
            gray_p2[:point] + gray_p1[point:]
          ]
          binaries_to_validate = [
            sub_strings_by_array(binary_children[0], bits),
            sub_strings_by_array(binary_children[1], bits),
            sub_strings_by_array(gray_children[0], bits),
            sub_strings_by_array(gray_children[1], bits)
          ]
          are_binaries_valid = self._validate_binaries_in_range(
            binaries_to_validate
          )

          if are_binaries_valid or attempts >= total_bits:
            break

          attempts += 1

        # binary_child = choice(binary_children)
        # gray_child = gray_children[binary_children.index(binary_child)]
        children.append(
          Individual(binary_children[0], gray_children[0], 0, bits)
        )
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
            mutated_children_length = len(mutated_children)
            mutated_children = mutated_children[:mutated_children_length - 1] \
              + [mutated_child]

            if (self._print):
              print(f'  Mutation for child completed: {mutated_child}\n')

            break

    if self._print:
      print(f'\nPopulation children after mutation: {mutated_children}\n')
      print()

    return mutated_children

  def _fitness(self, population_sample: List[Individual], minimize = True) -> None:
    """
    Method that calculates the fitness genotype of a given function for the
    current population.

    Args:
      population_sample (List[Individual]): a subset from the current population
      to calculate its fitness.

    Returns:
      float: total fitness from sample.
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
        fen = float(binary_to_float(gen, _range, self._precision)['number'])
        fens.append(fen)

      if (self._print):
        print(f'  gens: {gens}')
        print(f'  fens: {fens}')

      function: exp = sympify(str(self._function))

      for i, v in enumerate(variables_array):
        function = function.subs(v, fens[i])

      function_evaluations.append(function)

      if self._print:
        print(f'  fitness: {function}\n')

    if minimize:
      maxi = max(function_evaluations)

      for i, e in enumerate(function_evaluations):
        population_sample[i].set_score(1e-3 + maxi - e)
      pass
    else:
      mini = min(function_evaluations)

      for i, e in enumerate(function_evaluations):
        population_sample[i].set_score(1e-3 + e - mini)

  def canonical_algorithm(self, SAMPLE_SIZE: int, GEN_NUMBER = 200):
    self._select_initial_population(SAMPLE_SIZE)
    self._fitness(self._current_population)
    self._current_population.sort(reverse = True, key = lambda x: x.get_score())
    self._best_individual = self._current_population[0]

    for i in range(GEN_NUMBER):
      print(f'{i + 1}º iteration, best individual: {self._best_individual}')
      print(f'{i + 1}º iteration, population size: {len(self._current_population)}')

      children = self._crossover_one_point()
      mutated_children = self._mutation(children)
      self._fitness(mutated_children)
      self._update_current_population(mutated_children)
      self._current_population.sort(
        reverse = True,
        key = lambda x: x.get_score()
      )
      self._best_individual = self._current_population[0]

    print(f'{GEN_NUMBER + 1}º iteration, best individual: {self._best_individual}')
    print(f'{GEN_NUMBER + 1}º iteration, population size: {len(self._current_population)}')
