from typing import Dict, List, Sequence, Tuple, Union

from sympy import exp

from evo_tools.bin_gray import binary_to_float
from evo_tools.helpers import sub_strings_by_array
from evo_tools.models import Individual, SubPopulation

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

def _numbers_repr_from_decoded_strings(decoded_strings: List[str]) -> str:
  return '[' + ', '.join(decoded_strings) + ']'

def chromosome_to_numbers_repr(
  binary_or_gray: str,
  bits: Sequence[int],
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> str:
  binaries = sub_strings_by_array(binary_or_gray, bits)
  decoded_strings, _ = decode_binary_segments(binaries, sub_populations, precision)

  return _numbers_repr_from_decoded_strings(decoded_strings)

def build_individual(
  binary: str,
  gray: str,
  bits: Sequence[int],
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
    variables_array
  )

def build_individual_if_valid(
  binary: str,
  gray: str,
  bits: Sequence[int],
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> Union[Individual, None]:
  binaries = sub_strings_by_array(binary, bits)

  try:
    decoded_strings, _ = decode_binary_segments(
      binaries,
      sub_populations,
      precision
    )
  except Exception:
    return None

  return build_individual(
    binary,
    gray,
    bits,
    sub_populations,
    precision,
    parsed_function,
    variables_array,
    numbers_repr = _numbers_repr_from_decoded_strings(decoded_strings)
  )

def decode_individual(
  individual: Individual,
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> Tuple[List[str], List[float]]:
  binaries = sub_strings_by_array(
    individual.get_binary(),
    individual.get_bits_tuple()
  )
  try:
    _, decoded_numbers = decode_binary_segments(
      binaries,
      sub_populations,
      precision
    )
  except Exception:
    return binaries, []

  return binaries, decoded_numbers

def evaluate_function(
  parsed_function,
  variables_array: List[str],
  values: Sequence[float]
) -> exp:
  function = parsed_function

  for i, variable in enumerate(variables_array):
    function = function.subs(variable, values[i])

  return function

def build_solution(
  floats: Sequence[float],
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

def evaluate_individual_objective(
  individual: Individual,
  index: int,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  objective_function,
  should_print: bool = False
) -> Union[float, None]:
  chromosome = individual.get_binary()
  decoded_numbers = individual.get_numbers_tuple()

  if should_print:
    print(f'Chromosome {index}: {chromosome}')
    binaries, decoded_numbers = decode_individual(
      individual,
      sub_populations,
      precision
    )

  if should_print:
    print(f'  gens: {binaries}')
    print(f'  fens: {decoded_numbers}')

  if should_print and len(binaries) != len(decoded_numbers):
    if should_print:
      print(f'  fitness: Fail\n')

    return None

  objective_value = float(
    objective_function(*decoded_numbers)
  )

  if should_print:
    print(f'  fitness: {objective_value}\n')

  return objective_value
