from random import choices, sample
from typing import List, Tuple, Union

from evo_tools.bin_gray import get_binary_from_custom_representation, \
  get_float_from_custom_representation, get_gray_from_custom_representation
from evo_tools.models import Individual, SubPopulation
from evo_tools.phenotype import build_individual
from evo_tools.reporting import print_initial_population


def sample_sub_populations(
  sub_populations: List[SubPopulation],
  sample_size: int
) -> List[Tuple[List[str], int]]:
  return [
    (
      sample(sub_population.numbers, sample_size)
      if sample_size <= len(sub_population.numbers) else
      choices(sub_population.numbers, k = sample_size),
      sub_population.bits
    )
    for sub_population in sub_populations
  ]

def build_initial_individual(
  samples: List[Tuple[List[str], int]],
  index: int,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> Individual:
  selected_numbers = [current_sample[index] for current_sample, _ in samples]
  bits = [current_bits for _, current_bits in samples]
  binary = ''.join(
    get_binary_from_custom_representation(number)
    for number in selected_numbers
  )
  gray = ''.join(
    get_gray_from_custom_representation(number)
    for number in selected_numbers
  )
  numbers = '[' + ', '.join(
    get_float_from_custom_representation(number)
    for number in selected_numbers
  ) + ']'

  return build_individual(
    binary,
    gray,
    bits,
    sub_populations,
    precision,
    parsed_function,
    variables_array,
    numbers
  )

def select_initial_population(
  initial_population: List[Individual],
  sample_size: int,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str],
  should_print: bool = False
) -> Tuple[List[Individual], List[Individual]]:
  if len(initial_population) > 0:
    if should_print:
      print_initial_population(initial_population)

    current_population = initial_population.copy()
    return initial_population.copy(), current_population

  samples = sample_sub_populations(sub_populations, sample_size)
  first_sample, _ = samples[0]
  current_population = [
    build_initial_individual(
      samples,
      i,
      sub_populations,
      precision,
      parsed_function,
      variables_array
    )
    for i, _ in enumerate(first_sample)
  ]

  if should_print:
    print_initial_population(current_population)

  return current_population.copy(), current_population.copy()
