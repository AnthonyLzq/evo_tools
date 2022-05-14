from math import log2
# from random import sample
from typing import List, Tuple, Union

from custom import custom_range
from bin_gray import binary_numbers_with_n_bits, gray_numbers_with_n_bits, mutate_binary_or_gray, NumberBinaryAndGray, range_of_numbers_binary_and_gray

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

class Population():
  def __init__(
    self,
    ranges: List[Tuple[Union[float, int], Union[float, int]]],
    precision: Union[float, int],
    print: bool = False
  ) -> None:
    self.population_members: List[PopulationMember] = []
    self._print = print

    for rng in ranges:
      population_range, bits = range_of_numbers_binary_and_gray(rng, precision)
      self.population_members.append(
        PopulationMember(rng, population_range, bits)
      )

  def print(self):
    for population_member in self.population_members:
      print(f'Range:')
      print(population_member.rng)
      print()
      print(f'Bits:')
      print(population_member.bits)
      print()
      print(f'Numbers:')
      print(population_member.numbers)
      print()

  def select_initial_data(self, sample_size: int):
    # if (sample_size > len(self.binaries)):
    #   raise Exception(f'Sample size too big, maximum is: {len(self.binaries)}')

    # try:
    #   if (self._print):
    #     print(f'initial binary data: {self.initial_binary_data}')
    #     print(f'initial gray data  : {self.initial_gray_data}')

    #   return self.initial_binary_data, self.initial_gray_data
    # except:
    #   self.initial_binary_data = sample(self.binaries, sample_size)
    #   self.initial_gray_data = [
    #     self.grays[
    #       self.binaries.index(binary)
    #     ] for binary in self.initial_binary_data
    #   ]

    #   if (self._print):
    #     print(f'initial binary data: {self.initial_binary_data}')
    #     print(f'initial gray data  : {self.initial_gray_data}')

    #   return self.initial_binary_data, self.initial_gray_data
    pass

  def select(self, sample_size: int):
    # if (sample_size > len(self.binaries)):
    #   raise Exception(f'Sample size too big, maximum is: {len(self.binaries)}')

    # self.binary_selection = sample(self.binaries, sample_size)
    # self.gray_selection = [
    #   self.grays[
    #     self.binaries.index(binary)
    #   ] for binary in self.binary_selection
    # ]

    # if (self._print):
    #   print(f'binary selection: {self.binary_selection}')
    #   print(f'gray selection  : {self.gray_selection}')
    #   print()

    # return self.binary_selection, self.gray_selection
    pass

  def crossover(self, points: Tuple[int, int], from_initial: bool = False):
    # if (self._print):
    #   print('\nCrossover:\n')

    # p1, p2 = points

    # if p1 > self.bits - 1 or p2 > self.bits - 1:
    #   if p1 > self.bits - 1:
    #     raise Exception(f'Point {p1} out of range, maximum is: {self.bits - 3}')

    #   if p2 > self.bits - 1:
    #     raise Exception(f'Point {p2} out of range, maximum is: {self.bits - 1}')

    # binary_selection, gray_selection = self.select_initial_data(2) \
    #   if from_initial \
    #   else self.select(2)

    # binary_parent_1, binary_parent_2 = sample(binary_selection, 2)
    # gray_parent_1, gray_parent_2 = sample(gray_selection, 2)

    # binary_children = [
    #   binary_parent_1[:p1] + binary_parent_2[p1:p2] + binary_parent_1[p2:],
    #   binary_parent_2[:p1] + binary_parent_1[p1:p2] + binary_parent_2[p2:]
    # ]
    # gray_children = [
    #   gray_parent_1[:p1] + gray_parent_2[p1:p2] + gray_parent_1[p2:],
    #   gray_parent_2[:p1] + gray_parent_1[p1:p2] + gray_parent_2[p2:]
    # ]

    # if (self._print):
    #   print(f'binary parents : {[binary_parent_1, binary_parent_2]}')
    #   print(f'binary part 1  : {binary_parent_1[:p1]} + {binary_parent_2[p1:p2]} + {binary_parent_1[p2:]}')
    #   print(f'binary part 2  : {binary_parent_2[:p1]} + {binary_parent_1[p1:p2]} + {binary_parent_2[p2:]}')
    #   print(f'binary children: {binary_children}')
    #   print()
    #   print(f'gray parents : {[gray_parent_1, gray_parent_2]}')
    #   print(f'gray part 1  : {gray_parent_1[:p1]} + {gray_parent_2[p1:p2]} + {gray_parent_1[p2:]}')
    #   print(f'gray part 2  : {gray_parent_2[:p1]} + {gray_parent_1[p1:p2]} + {gray_parent_2[p2:]}')
    #   print(f'gray children: {gray_children}')

    # return binary_children, gray_children
    pass

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
