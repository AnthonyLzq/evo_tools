from random import sample, random, randint
from math import log
from functools import reduce
from sympy import exp
from typing import List, Tuple, Union
from sys import version_info

if version_info >= (3, 8):
  from typing import TypedDict
else:
  from typing_extensions import TypedDict

from evo_tools.bin_gray import NumberBinaryAndGray, binary_to_float, mutate_binary_or_gray, range_of_numbers_binary_and_gray
from evo_tools.helpers import sub_strings_by_array

class Sample(TypedDict):
  """
  A custom dict model which represents a sample.
  """
  binaries: List[str]
  grays: List[str]
  bits: List[int]

class PopulationMember():
  """
  A class to represent a PopulationMember
  --

  A PopulationMember is nothing but a object that represents a real range (float interval).
  So, a Population is build with several ranges, with its representation in binary
  and gray code and the number of bits that are used to represent the range.

  For example, lets say you want to create a Population of one range: [1, 2],
  with a precision of 0.1, then we will only have an array of PopulationMember,
  whose len will be one, and that only member will store its class attributes as follows:

  Attributes
  --

  rng: Tuple[Union[float, int], Union[float, int]]
    The range specified for this PopulationMember, for this case (1, 2)

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
    return f'"rng": {self.rng}, "numbers": {self.numbers}, "bits": {self.bits}'

class Population():
  """
  A class to represent a population
  --

  It is conformed by several PopulationMembers.

  Attributes
  --

  _population_members: List[:class:`PopulationMember`]
    A list of every PopulationMember from the Population, which is defined by
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

    self._population_members: List[PopulationMember] = []
    self._precision = precision
    self._crossover_rate = crossover_rate
    self._mutation_rate = mutation_rate
    self._variables = variables
    self._function = function
    self._print = _print

    p10 = pow(precision, -1) if precision != 1 else 1
    self._n_decimal_digits = int(round(log(p10, 10)))

    for rng in ranges:
      population_range, bits = range_of_numbers_binary_and_gray(
        rng,
        self._precision
      )
      self._population_members.append(
        PopulationMember(rng, population_range, bits)
      )

    self._max_sample_size = len(self._population_members[0].numbers)
    variables_array = self._variables.split()

    if (len(variables_array) != len(self._population_members)):
      raise Exception('Variables size does not match the number of ranges')

    for population_member in self._population_members:
      aux = len(population_member.numbers)

      if aux < self._max_sample_size:
        self._max_sample_size = aux

  def print(self):
    """
    Prints the current Population data. The current sample and the data from each
    PopulationMember.
    """
    print('\nCurrent population sample:\n')
    print(self._current_data)
    print('\nData from population members:')

    for population_member in self._population_members:
      print('\nRange:\n')
      print(population_member.rng)
      print()
      print('\nBits:\n')
      print(population_member.bits)
      print()
      print('\nNumbers:\n')
      print(population_member.numbers)
      print()

  def select_initial_data(self, sample_size: int) -> Sample:
    """
    Method that selects the initial sample (randomly) of the Population.

    Args:
      sample_size: int
        Population sample size.

    Raises:
      Exception: When the required sample_size is bigger than the maximum sample
      size (the lowest range size from the domain).

    Returns:
      Sample: A dict with a binaries, grays and floating numbers list, all of them
      represented with strings: :class:`Sample`
    """
    self._sample_size = sample_size

    if (self._sample_size > self._max_sample_size):
      raise Exception(
        f'Sample size too big, maximum is: {self._max_sample_size}'
      )

    try:
      if self._print:
        print('\nInitial data:\n')
        print(self._initial_data)

      return self._initial_data
    except:
      samples: List[List[NumberBinaryAndGray]] = []
      bits = []
      binaries = []
      grays = []

      for population_member in self._population_members:
        samples.append(
          sample(population_member.numbers, sample_size)
        )

      f_sample = samples[0]

      for i, __ in enumerate(f_sample):
        binary = ''
        gray = ''

        for j, _ in enumerate(self._population_members):
          binary += samples[j][i]['binary']
          gray += samples[j][i]['gray']

        binaries.append(binary)
        grays.append(gray)

      for population_member in self._population_members:
        bits.append(population_member.bits)

      self._initial_data: Sample = {
        'binaries': binaries,
        'grays': grays,
        'bits': bits
      }
      self._current_data = self._initial_data.copy()

      return self._current_data.copy()

  def get_current_data(self):
    """
    Returns a copy of the current Population data.

    Returns: :class:`Sample`
    """
    return self._current_data.copy()

  def get_sample_from_data(self, sample_size: int) -> Sample:
    """
    Method that selects a new sample randomly.

    Args:
      sample_size: int

    Returns: :class:`Sample`
    """
    current_binaries = self._current_data['binaries']
    current_grays = self._current_data['grays']
    bits = self._current_data['bits']

    return {
      'binaries': sample(current_binaries, sample_size),
      'grays': sample(current_grays, sample_size),
      'bits': bits
    }

  def update_current_data(self, binaries: List[str], grays: List[str]) -> None:
    """
    Method that updates the current sample, after crossover or mutation.

    Args:
      binaries: List[str]
        New list of binaries
      grays List[str]
        New list of grays
    """
    self._current_data: Sample = {
      'binaries': binaries,
      'grays': grays,
      'bits': self._current_data['bits']
    }

  def select(self, sample_size: int) -> None:
    """
    Method that creates a new sample and update the current sample with the new one.

    Args:
      sample_size:int

    Raises:
      Exception: when the sample size is too big or the initial data was not selected.
    """

    if (sample_size > self._max_sample_size):
      raise Exception(
        f'Sample size too big, maximum is: {self._max_sample_size}'
      )

    try:
      sample_data = self.get_sample_from_data(sample_size)
      self.update_current_data(sample_data['binaries'], sample_data['grays'])

      if self._print:
        print('\nSelection: \n')
        print(self._current_data)
    except:
      raise Exception(
        'Select initial data was not invoked at the beginning. It must be.'
      )

  def validate_binaries_in_range(self, binaries: List[List[str]]) -> bool:
    """
    Method that validates if a given list of binaries are in the domain.

    Args:
      binaries: List[List[str]]
        List genotypes from each PopulationMember.

    Returns:
      bool: whether or not binaries are valid
    """
    for b in binaries:
      for i, gen in enumerate(b):
        try:
          _range = self._population_members[i].rng
          fen = binary_to_float(gen, _range, self._precision)
          x0, xf = _range

          if float(fen['number']) < x0 or float(fen['number']) > xf:
            return False
        except:
          return False

    return True

  def crossover_one_point(self) -> None:
    """
    Method that creates 2 children from 2 parents combining their genotype
    and the crossover probability.

    Raises:
      Exception: when the initial data wasn't selected
    """
    p = random()

    if random() < self._crossover_rate:
      if (self._print):
        print('\nCrossover: \n')

      total_bits = 0
      bits = []

      try:
        bits = self._current_data['bits']
        total_bits = reduce(lambda a, b: a + b, self._current_data['bits'])
        point = randint(0, total_bits - 1)
        binaries = self._current_data['binaries']
        grays = self._current_data['grays']

        while True:
          binary_parent_1, binary_parent_2 = sample(binaries, 2)

          binary_children = [
            binary_parent_1[:point] + binary_parent_1[point:],
            binary_parent_2[:point] + binary_parent_2[point:]
          ]

          binaries_to_validate = [
            sub_strings_by_array(binary_children[0], bits),
            sub_strings_by_array(binary_children[1], bits)
          ]
          are_binaries_valid = self.validate_binaries_in_range(
            binaries_to_validate
          )

          if are_binaries_valid:
            break

        gray_parent_1 = grays[binaries.index(binary_parent_1)]
        gray_parent_2 = grays[binaries.index(binary_parent_2)]

        gray_children = [
          gray_parent_1[:point] + gray_parent_1[point:],
          gray_parent_2[:point] + gray_parent_2[point:]
        ]

        if (self._print):
          print(f'binary parents : {[binary_parent_1, binary_parent_2]}')
          print(f'binary part 1  : {binary_parent_1[:point]} + {binary_parent_1[point:]}')
          print(f'binary part 2  : {binary_parent_2[:point]} + {binary_parent_2[point:]}')
          print(f'binary children: {binary_children}')
          print()
          print(f'gray parents : {[gray_parent_1, gray_parent_2]}')
          print(f'gray part 1  : {gray_parent_1[:point]} + {gray_parent_1[point:]}')
          print(f'gray part 2  : {gray_parent_2[:point]} + {gray_parent_2[point:]}')
          print(f'gray children: {gray_children}')

        binaries += binary_children
        grays += gray_children

        self.update_current_data(binaries, grays)
      except:
        raise Exception(
          'Select initial data was not invoked at the beginning. It must be.'
        )
    elif self._print:
      print(f'Crossover failed because p = {p} < {self._crossover_rate}')

  def mutation(self) -> None:
    """
    Method that changes 1 bit from a children based in the mutation probability.

    Raises:
      Exception: when the initial data wasn't selected.
    """
    p = random()

    if p < self._mutation_rate:
      if (self._print):
        print('\nMutation: \n')

      try:
        binaries = self._current_data['binaries']
        grays = self._current_data['grays']

        if (self._print):
          print(f'binaries before mutation: {binaries}')
          print(f'grays before mutation: {grays}')
          print()

        binary_selected = sample(binaries, 1)[0]
        index = binaries.index(binary_selected)
        gray_selected = grays[index]

        binaries = binaries[:index] \
          + [mutate_binary_or_gray(binary_selected)] \
          + binaries[index + 1:]
        grays = grays[:index] \
          + [mutate_binary_or_gray(gray_selected)] \
          + grays[index + 1:]

        self.update_current_data(binaries, grays)
      except:
        raise Exception(
          'Select initial data was not invoked at the beginning. It must be.'
        )
    elif self._print:
      print(f'Crossover failed because p = {p} < {self._crossover_rate}')

  def fitness(self):
    """
    Method that calculates the fitness genotype of a given function for the
    current population.
    """
    variables_array = self._variables.split()
    binaries = self._current_data['binaries']
    # grays = self._current_data['grays']
    bits = self._current_data['bits']

    for i, chromosome in enumerate(binaries):
      if (self._print):
        print(f'Chromosome {i}: {chromosome}')

      gens = sub_strings_by_array(chromosome, bits)
      fens: List[float] = []

      for i, gen in enumerate(gens):
        _range = self._population_members[i].rng
        fen = float(binary_to_float(gen, _range, self._precision)['number'])
        fens.append(fen)

      if (self._print):
        print(f'gens: {gens}')
        print(f'fens: {fens}')

      fitness = self._function.copy()

      for i, v in enumerate(variables_array):
        fitness = fitness.subs(v, fens[i])

      final_fitness = format(fitness, f'.{self._n_decimal_digits}f')
      print(f'fitness: {final_fitness}')
      print()

  def canonical_algorithm(self):
    pass
