from random import sample
from math import log
from typing import List, Tuple,TypedDict, Union
from functools import reduce
from sympy import symbols, exp

from bin_gray import NumberBinaryAndGray, binary_to_float, mutate_binary_or_gray, range_of_numbers_binary_and_gray
from helpers import sub_strings_by_array

class Sample(TypedDict):
  binaries: List[str]
  grays: List[str]
  bits: List[int]

class PopulationMember():
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
  def __init__(
    self,
    ranges: List[Tuple[Union[float, int], Union[float, int]]],
    precision: Union[float, int],
    _print: bool = False
  ) -> None:
    self._population_members: List[PopulationMember] = []
    self._precision = precision
    self._print = _print

    p10 = pow(precision, -1) if precision != 1 else 1
    self._n_decimal_digits = int(round(log(p10, 10)))

    if len(ranges) == 0:
      raise Exception('At least one range is required')

    for rng in ranges:
      population_range, bits = range_of_numbers_binary_and_gray(
        rng,
        self._precision
      )
      # self._total_numbers += population_range
      self._population_members.append(
        PopulationMember(rng, population_range, bits)
      )

    self.max_sample_size = len(self._population_members[0].numbers)

    for population_member in self._population_members:
      aux = len(population_member.numbers)

      if aux < self.max_sample_size:
        self.max_sample_size = aux

  def print(self):
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
    self._sample_size = sample_size

    if (self._sample_size > self.max_sample_size):
      raise Exception(
        f'Sample size too big, maximum is: {self.max_sample_size}'
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

  def get_sample_from_data(self, sample_size: int) -> Sample:
    current_binaries = self._current_data['binaries']
    current_grays = self._current_data['grays']
    bits = self._current_data['bits']

    return {
      'binaries': sample(current_binaries, sample_size),
      'grays': sample(current_grays, sample_size),
      'bits': bits
    }

  def update_current_data(self, binaries: List[str], grays: List[str]) -> None:
    self._current_data: Sample = {
      'binaries': binaries,
      'grays': grays,
      'bits': self._current_data['bits']
    }

  def select(self, sample_size: int) -> None:
    if (sample_size > self.max_sample_size):
      raise Exception(
        f'Sample size too big, maximum is: {self.max_sample_size}'
      )

    try:
      sample_data = self.get_sample_from_data(sample_size)
      self.update_current_data(sample_data['binaries'], sample_data['grays'])

      if self._print:
        print('\nSelection: \n')
        print(self._current_data)
    except:
      raise Exception(
        'Select initial data was not invoked at the beging. It must be.'
      )

  def validate_binaries_in_range(self, binaries: List[List[str]]) -> bool:
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

  def crossover(self, points: Tuple[int, int]) -> None:
    if (self._print):
      print('\nCrossover: \n')

    p1, p2 = points
    total_bits = 0
    bits = []

    try:
      bits = self._current_data['bits']
      total_bits = reduce(lambda a, b: a + b, self._current_data['bits'])

      if p1 > total_bits - 1 or p2 > total_bits - 1:
        if p1 > total_bits - 1:
          raise Exception(
            f'Point {p1} out of range, maximum is: {total_bits - 3}'
          )

        if p2 > self.total_bits - 1:
          raise Exception(
            f'Point {p2} out of range, maximum is: {total_bits - 1}'
          )

      binaries = self._current_data['binaries']
      grays = self._current_data['grays']

      while True:
        binary_parent_1, binary_parent_2 = sample(binaries, 2)

        binary_children = [
          binary_parent_1[:p1] + binary_parent_2[p1:p2] + binary_parent_1[p2:],
          binary_parent_2[:p1] + binary_parent_1[p1:p2] + binary_parent_2[p2:]
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
        gray_parent_1[:p1] + gray_parent_2[p1:p2] + gray_parent_1[p2:],
        gray_parent_2[:p1] + gray_parent_1[p1:p2] + gray_parent_2[p2:]
      ]

      if (self._print):
        print(f'binary parents : {[binary_parent_1, binary_parent_2]}')
        print(f'binary part 1  : {binary_parent_1[:p1]} + {binary_parent_2[p1:p2]} + {binary_parent_1[p2:]}')
        print(f'binary part 2  : {binary_parent_2[:p1]} + {binary_parent_1[p1:p2]} + {binary_parent_2[p2:]}')
        print(f'binary children: {binary_children}')
        print()
        print(f'gray parents : {[gray_parent_1, gray_parent_2]}')
        print(f'gray part 1  : {gray_parent_1[:p1]} + {gray_parent_2[p1:p2]} + {gray_parent_1[p2:]}')
        print(f'gray part 2  : {gray_parent_2[:p1]} + {gray_parent_1[p1:p2]} + {gray_parent_2[p2:]}')
        print(f'gray children: {gray_children}')

      binaries += binary_children
      grays += gray_children

      self.update_current_data(binaries, grays)
    except:
      raise Exception(
        'Select initial data was not invoked at the beging. It must be.'
      )

  def mutation(self) -> None:
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
        'Select initial data was not invoked at the beging. It must be.'
      )

  def fitness(self, variables: str, f: exp):
    variables_array = variables.split()

    if (len(variables_array) != len(self._population_members)):
      raise Exception('Variables size does not match the number of ranges')

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

      fitness = f.copy()
      for i, v in enumerate(variables_array):
        fitness = fitness.subs(v, fens[i])

      final_fitness = format(fitness, f'.{self._n_decimal_digits}f')
      print(f'fitness: {final_fitness}')
      print()

  def canonical_algorithm(self):
    pass
