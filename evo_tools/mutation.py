from typing import List, Union

from evo_tools.bin_gray import binary_to_gray, format_to_n_bits, \
  mutate_n_bits_from_binary_or_gray, mutation_binary_or_gray_by_flipping
from evo_tools.helpers import sub_strings_by_array
from evo_tools.models import Individual, SubPopulation
from evo_tools.phenotype import build_individual, validate_binaries_in_range

def apply_mutation(
  mutation_method: str,
  binary_or_gray: str
) -> str:
  if mutation_method == 'one_point':
    return mutate_n_bits_from_binary_or_gray(binary_or_gray)

  if mutation_method == 'two_points':
    return mutate_n_bits_from_binary_or_gray(binary_or_gray, 2)

  if mutation_method == 'flipping':
    return mutation_binary_or_gray_by_flipping(binary_or_gray)

  raise Exception('Mutation method not allowed')

def validate_mutation_method(mutation_method: str) -> None:
  if mutation_method in ['one_point', 'two_points', 'flipping']:
    return

  raise Exception('Mutation method not allowed')


def mutate_individual(
  child: Individual,
  mutation_method: str,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> Union[Individual, None]:
  attempts = 0
  bits = child.get_bits()

  while True:
    binary = apply_mutation(
      mutation_method,
      child.get_binary()
    )
    gray = format_to_n_bits(
      binary_to_gray(binary),
      sum(bits)
    )
    binaries_to_validate = [
      sub_strings_by_array(binary, bits),
      sub_strings_by_array(gray, bits)
    ]
    are_binaries_valid = validate_binaries_in_range(
      binaries_to_validate,
      sub_populations,
      precision
    )

    if are_binaries_valid:
      return build_individual(
        binary,
        gray,
        bits,
        sub_populations,
        precision,
        parsed_function,
        variables_array
      )

    if attempts > 5:
      return None

    attempts += 1
