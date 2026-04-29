from typing import Dict, List, Tuple, Union

from sympy import exp

from evo_tools.bin_gray import binary_to_float
from evo_tools.helpers import sub_strings_by_array
from evo_tools.models import Individual, SubPopulation

def validate_binaries_in_range(
  binaries: List[List[str]],
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> bool:
  for binary in binaries:
    for i, gen in enumerate(binary):
      try:
        binary_to_float(
          gen,
          sub_populations[i].numbers_dict,
          sub_populations[i].rng,
          precision
        )
      except Exception:
        return False

  return True

def decode_binary_segments(
  binaries: List[str],
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> Tuple[List[str], List[float]]:
  decoded_strings: List[str] = []
  decoded_numbers: List[float] = []

  for i, binary in enumerate(binaries):
    decoded_string = binary_to_float(
      binary,
      sub_populations[i].numbers_dict,
      sub_populations[i].rng,
      precision
    )
    decoded_strings.append(decoded_string)
    decoded_numbers.append(float(decoded_string))

  return decoded_strings, decoded_numbers

def chromosome_to_numbers_repr(
  binary_or_gray: str,
  bits: List[int],
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> str:
  binaries = sub_strings_by_array(binary_or_gray, bits)
  decoded_strings, _ = decode_binary_segments(binaries, sub_populations, precision)

  return '[' + ', '.join(decoded_strings) + ']'

def build_individual(
  binary: str,
  gray: str,
  bits: List[int],
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str],
  numbers_repr: Union[str, None] = None
) -> Individual:
  if numbers_repr is None:
    numbers_repr = chromosome_to_numbers_repr(
      binary,
      bits,
      sub_populations,
      precision
    )

  return Individual(
    binary,
    gray,
    0,
    bits,
    numbers_repr,
    parsed_function,
    variables_array.copy()
  )

def decode_individual(
  individual: Individual,
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> Tuple[List[str], List[float]]:
  binaries = sub_strings_by_array(
    individual.get_binary(),
    individual.get_bits()
  )
  decoded_numbers: List[float] = []

  if validate_binaries_in_range([binaries], sub_populations, precision):
    _, decoded_numbers = decode_binary_segments(
      binaries,
      sub_populations,
      precision
    )

  return binaries, decoded_numbers

def evaluate_function(
  parsed_function,
  variables_array: List[str],
  values: List[float]
) -> exp:
  function = parsed_function

  for i, variable in enumerate(variables_array):
    function = function.subs(variable, values[i])

  return function

def build_solution(
  floats: List[float],
  parsed_function,
  variables_array: List[str]
) -> Tuple[Dict[str, float], exp]:
  if len(floats) == 0:
    raise Exception('Something went wrong')

  function = evaluate_function(parsed_function, variables_array, floats)
  solution: Dict[str, float] = {}

  for i, variable in enumerate(variables_array):
    solution[variable] = floats[i]

  return solution, function
