from math import log2
from random import sample
from typing import List, Tuple, Union

from evo_tools.custom import custom_range
from evo_tools.bin_gray import binary_numbers_with_n_bits, gray_numbers_with_n_bits, mutate_binary_or_gray

class PopulationMember():
  def __init__(
    self,
    general_interval: List[str],
    binaries: List[str],
    grays: List[str],
    positional_range: Tuple[Union[float, int], Union[float, int]]
  ) -> None:
    pos_x0, pos_xf = positional_range
    self.interval = general_interval[
      pos_x0:pos_xf + 1
    ]
    self.binaries = binaries[
      pos_x0:pos_xf + 1
    ]
    self.grays = grays[
      pos_x0:pos_xf + 1
    ]

class Population():
  def __init__(
    self,
    ranges: List[Tuple[Union[float, int], Union[float, int]]],
    decimals: int,
    print: bool = False
  ) -> None:
    self.population_members: List[PopulationMember] = []
    self._print = print

    for range in ranges:
      x0, xf = range
      current_bits = int(log2((xf - x0) * pow(10, decimals)) + 0.9)

      if current_bits > bits:
        bits = current_bits

      if lower_x0 > x0:
        lower_x0 = x0

      if upper_xf < xf:
        upper_xf = xf

      general_range = (lower_x0, upper_xf)

    g_x0, g_xf = general_range
    self.general_interval = [
      round(
        i,
        decimals
      ) for i in custom_range(
        g_x0,
        g_xf + pow(10, -decimals),
        pow(10, -decimals)
      )
    ]
    self.g_length = len(self.general_interval)
    self.bits = bits

    if self.general_interval[self.g_length - 1] > g_xf:
      self.general_interval.pop()
      self.g_length -= 1

    binaries = binary_numbers_with_n_bits(self.g_length, bits)
    grays = gray_numbers_with_n_bits(self.g_length, bits)

    for range in ranges:
      x0, xf = range
      pos_x0 = self.general_interval.index(x0)
      pos_xf = self.general_interval.index(xf)

      self.population_members.append(
        PopulationMember(
          self.general_interval,
          binaries,
          grays,
          (pos_x0, pos_xf)
        )
      )

    self.intervals: List[List[str]] = []
    self.binaries_intervals: List[List[str]] = []
    self.grays_intervals: List[List[str]] = []
    self.binaries: List[str] = []
    self.grays: List[str] = []

    for member in self.population_members:
      self.intervals += [member.interval]
      self.binaries_intervals += [member.binaries]
      self.grays_intervals += [member.grays]
      self.binaries += member.binaries
      self.grays += member.grays

    self.binaries = list(dict.fromkeys(self.binaries))
    self.grays = list(dict.fromkeys(self.grays))

  def print_members_data(
    self,
    interval = False,
    binaries = False,
    grays = False
  ):
    for member in self.population_members:
      if interval:
        print(f'interval: {member.interval}')

      if binaries:
        print(f'binaries: {member.binaries}')

      if grays:
        print(f'grays: {member.grays}')

      if interval or binaries or grays:
        print()

  def print(self):
    print(f'intervals: {self.intervals}\n')
    print(f'binaries_intervals: {self.binaries_intervals}\n')
    print(f'grays_intervals: {self.grays_intervals}\n')
    print(f'binaries: {self.binaries}\n')
    print(f'grays: {self.grays}\n')
    print(f'bits: {self.bits}\n')

  def select_initial_data(self, sample_size: int):
    if (sample_size > len(self.binaries)):
      raise Exception(f'Sample size too big, maximum is: {len(self.binaries)}')

    try:
      if (self._print):
        print(f'initial binary data: {self.initial_binary_data}')
        print(f'initial gray data  : {self.initial_gray_data}')

      return self.initial_binary_data, self.initial_gray_data
    except:
      self.initial_binary_data = sample(self.binaries, sample_size)
      self.initial_gray_data = [
        self.grays[
          self.binaries.index(binary)
        ] for binary in self.initial_binary_data
      ]

      if (self._print):
        print(f'initial binary data: {self.initial_binary_data}')
        print(f'initial gray data  : {self.initial_gray_data}')

      return self.initial_binary_data, self.initial_gray_data

  def select(self, sample_size: int):
    if (sample_size > len(self.binaries)):
      raise Exception(f'Sample size too big, maximum is: {len(self.binaries)}')

    self.binary_selection = sample(self.binaries, sample_size)
    self.gray_selection = [
      self.grays[
        self.binaries.index(binary)
      ] for binary in self.binary_selection
    ]

    if (self._print):
      print(f'binary selection: {self.binary_selection}')
      print(f'gray selection  : {self.gray_selection}')
      print()

    return self.binary_selection, self.gray_selection

  def crossover(self, points: Tuple[int, int], from_initial: bool = False):
    if (self._print):
      print('\nCrossover:\n')

    p1, p2 = points

    if p1 > self.bits - 1 or p2 > self.bits - 1:
      if p1 > self.bits - 1:
        raise Exception(f'Point {p1} out of range, maximum is: {self.bits - 3}')

      if p2 > self.bits - 1:
        raise Exception(f'Point {p2} out of range, maximum is: {self.bits - 1}')

    binary_selection, gray_selection = self.select_initial_data(2) \
      if from_initial \
      else self.select(2)

    binary_parent_1, binary_parent_2 = sample(binary_selection, 2)
    gray_parent_1, gray_parent_2 = sample(gray_selection, 2)

    binary_children = [
      binary_parent_1[:p1] + binary_parent_2[p1:p2] + binary_parent_1[p2:],
      binary_parent_2[:p1] + binary_parent_1[p1:p2] + binary_parent_2[p2:]
    ]
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
    binary_selection, gray_selection = self.select_initial_data(1) \
      if from_initial \
      else self.select(1)

    binary, = binary_selection
    gray, = gray_selection

    binary_index = self.binaries.index(binary)
    gray_index = self.grays.index(gray)

    if (self._print):
      print(f'binaries before mutation: {self.binaries}')
      print(f'grays before mutation: {self.grays}')
      print()

    self.binaries = self.binaries[:binary_index] \
      + [mutate_binary_or_gray(binary)] \
      + self.binaries[binary_index + 1:]
    self.grays = self.grays[:gray_index] \
      + [mutate_binary_or_gray(gray)] \
      + self.grays[gray_index + 1:]

    if (self._print):
      print(f'binaries after mutation: {self.binaries}')
      print(f'grays after mutation: {self.grays}')
      print()
