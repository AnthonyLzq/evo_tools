from json import loads
from typing import Dict, List, Tuple, Union

from evo_tools.bin_gray import get_binary_from_custom_representation

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
    self._objective_value: Union[float, None] = None
    self._bits = bits.copy()
    self._total_bits = sum(self._bits)
    self._bits_repr = f'[{", ".join(str(bit) for bit in self._bits)}]'
    self._numbers = numbers
    self._numbers_array = loads(numbers)
    self._function = function
    self._variables_array = variables_array.copy()

  def get_binary(self) -> str:
    return self._binary

  def get_gray(self) -> str:
    return self._gray

  def get_score(self) -> float:
    return float(self._score)

  def get_bits(self) -> List[int]:
    return self._bits.copy()

  def get_total_bits(self) -> int:
    return self._total_bits

  def set_score(self, score: float) -> None:
    self._score = score

  def set_objective_value(self, objective_value: Union[float, None]) -> None:
    self._objective_value = objective_value

  def get_objective_value(self) -> float:
    if self._objective_value is None:
      raise Exception('Objective value has not been calculated')

    return self._objective_value

  def get_numbers(self) -> str:
    return self._numbers

  def get_fitness(self):
    if self._objective_value is not None:
      return self._objective_value

    f = self._function

    for i, n in enumerate(self._numbers_array):
      f = f.subs(self._variables_array[i], n)

    return float(f)

  def __str__(self) -> str:
    return f'{{ \
"binary": "{self._binary}", \
"gray": "{self._gray}", \
"numbers": "{self._numbers}", \
"bits": "{self._bits_repr}", \
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

  numbers: List[str]

  bits: int
    Number of bits used for represent the float value.
  """
  def __init__(
    self,
    rng: Tuple[Union[float, int], Union[float, int]],
    numbers: List[str],
    bits: int,
  ) -> None:
    self.rng = rng
    self.numbers = numbers
    self.bits = bits
    self.numbers_dict: Dict[str, str] = {}

    for n in self.numbers:
      self.numbers_dict[get_binary_from_custom_representation(n)] = n

  def __str__(self) -> str:
    return f'{{ "rng": {self.rng}, "numbers": {self.numbers}, "bits": {self.bits} }}'
