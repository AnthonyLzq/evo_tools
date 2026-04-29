from typing import List, Tuple, Union

from evo_tools.bin_gray import range_of_numbers_binary_and_gray
from evo_tools.models import SubPopulation


def build_sub_populations(
  ranges: List[Tuple[Union[float, int], Union[float, int]]],
  precision: Union[float, int],
  sample_size: int
) -> List[SubPopulation]:
  sub_populations: List[SubPopulation] = []

  for rng in ranges:
    sub_population_range, bits = range_of_numbers_binary_and_gray(
      rng,
      precision,
      sample_size
    )
    sub_populations.append(
      SubPopulation(rng, sub_population_range, bits)
    )

  return sub_populations

def resolve_max_sample_size(sub_populations: List[SubPopulation]) -> int:
  max_sample_size = len(sub_populations[0].numbers)

  for sub_population in sub_populations:
    sub_population_size = len(sub_population.numbers)

    if sub_population_size < max_sample_size:
      max_sample_size = sub_population_size

  return max_sample_size

def validate_variable_count(
  variables_array: List[str],
  sub_populations: List[SubPopulation]
) -> None:
  if len(variables_array) != len(sub_populations):
    raise Exception('Variables size does not match the number of ranges')
