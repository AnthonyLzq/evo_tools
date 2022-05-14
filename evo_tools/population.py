from json import dumps
from random import sample
from typing import List, Tuple,TypedDict, Union
from functools import reduce

from bin_gray import NumberBinaryAndGray, binary_to_float, range_of_numbers_binary_and_gray
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
    for population_member in self._population_members:
      print(f'Range:')
      print(population_member.rng)
      print()
      print(f'Bits:')
      print(population_member.bits)
      print()
      print(f'Numbers:')
      print(population_member.numbers)
      print()

  def select_initial_data(self, sample_size: int) -> Sample:
    if (sample_size > self.max_sample_size):
      raise Exception(
        f'Sample size too big, maximum is: {self.max_sample_size}'
      )

    try:
      if self._print:
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

      return self._initial_data

  def update_current_data(self, sample_size: int):
    current_binaries = self._current_data['binaries']
    current_grays = self._current_data['grays']
    bits = self._current_data['bits']
    self._current_data = {
      'binaries': sample(current_binaries, sample_size),
      'grays': sample(current_grays, sample_size),
      'bits': bits
    }

    return self._current_data

  def select(self, sample_size: int):
    if (sample_size > self.max_sample_size):
      raise Exception(
        f'Sample size too big, maximum is: {self.max_sample_size}'
      )

    try:
      return self.update_current_data(sample_size)
    except:
      self.select_initial_data()

      return self.update_current_data(sample_size)

  def validate_binaries_in_range(self, binaries: List[List[str]]):
    for bins in binaries:
      for i, b in enumerate(bins):
        try:
          _range = self._population_members[i].rng
          f = binary_to_float(b, _range, self._precision)
          x0, xf = _range

          if float(f['number']) < x0 or float(f['number']) > xf:
            return False
        except:
          return False

    return True

  def crossover(self, points: Tuple[int, int]):
    if (self._print):
      print('\nCrossover: \n')

    p1, p2 = points
    total_bits = 0
    bits = []

    try:
      bits = self._current_data['bits']
      total_bits = reduce(lambda a, b: a + b, self._current_data['bits'])
    except:
      self.select_initial_data()
      bits = self._current_data['bits']
      total_bits = reduce(lambda a, b: a + b, self._current_data['bits'])
    finally:
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

      a = 10
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
        are_binaries_valid = self.validate_binaries_in_range(binaries_to_validate)
        print(are_binaries_valid)
        a -= 1
        if a < 0:
          raise Exception

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

      return binary_children, gray_children

  def mutation(self, from_initial: bool = False):
    # binary_selection, gray_selection = self.select_initial_data(1) \
    #   if from_initial \
    #   else self.select(1)

    # binary, = binary_selection
    # gray, = gray_selection

    # binary_index = self.binaries.index(binary)
    # gray_index = self.grays.index(gray)

    # if (self._print):
    #   print(f'binaries before mutation: {self.binaries}')
    #   print(f'grays before mutation: {self.grays}')
    #   print()

    # self.binaries = self.binaries[:binary_index] \
    #   + [mutate_binary_or_gray(binary)] \
    #   + self.binaries[binary_index + 1:]
    # self.grays = self.grays[:gray_index] \
    #   + [mutate_binary_or_gray(gray)] \
    #   + self.grays[gray_index + 1:]

    # if (self._print):
    #   print(f'binaries after mutation: {self.binaries}')
    #   print(f'grays after mutation: {self.grays}')
    #   print()
    pass
