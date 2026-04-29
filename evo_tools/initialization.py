from typing import List, Tuple, Union

from evo_tools.bin_gray import get_binary_from_custom_representation, \
  get_float_from_custom_representation, get_gray_from_custom_representation
from evo_tools.models import Individual, SubPopulation
from evo_tools.phenotype import build_individual
from evo_tools.randomness import sample_with_replacement, sample_without_replacement
from evo_tools.reporting import print_initial_population


def sample_sub_populations(
  sub_populations: List[SubPopulation],
  sample_size: int
) -> List[List[str]]:
  return [
    sample_without_replacement(sub_population.numbers, sample_size)
    if sample_size <= len(sub_population.numbers) else
    sample_with_replacement(sub_population.numbers, sample_size)
    for sub_population in sub_populations
  ]

def build_initial_individual(
  samples: List[List[str]],
  bits: List[int],
  index: int,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> Individual:
  selected_numbers = [current_sample[index] for current_sample in samples]
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

    return initial_population, initial_population.copy()

  samples = sample_sub_populations(sub_populations, sample_size)
  first_sample = samples[0]
  bits = [sub_population.bits for sub_population in sub_populations]
  current_population = [
    build_initial_individual(
      samples,
      bits,
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

  return current_population.copy(), current_population
