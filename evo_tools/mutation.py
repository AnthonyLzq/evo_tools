from random import random
from typing import List, Union

from evo_tools.bin_gray import binary_to_gray, format_to_n_bits, \
  mutate_n_bits_from_binary_or_gray, mutation_binary_or_gray_by_flipping
from evo_tools.models import Individual, SubPopulation
from evo_tools.phenotype import build_individual_if_valid

MUTATION_METHODS = (
  'one_point',
  'two_points',
  'flipping'
)


def apply_mutation(
  mutation_method: str,
  binary_or_gray: str
) -> str:
  mutation_functions = {
    MUTATION_METHODS[0]: lambda: mutate_n_bits_from_binary_or_gray(binary_or_gray),
    MUTATION_METHODS[1]: lambda: mutate_n_bits_from_binary_or_gray(binary_or_gray, 2),
    MUTATION_METHODS[2]: lambda: mutation_binary_or_gray_by_flipping(binary_or_gray)
  }

  try:
    return mutation_functions[mutation_method]()
  except KeyError as exc:
    raise Exception('Mutation method not allowed') from exc

def validate_mutation_method(mutation_method: str) -> None:
  if mutation_method in MUTATION_METHODS:
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
  total_bits = child.get_total_bits()

  while True:
    binary = apply_mutation(
      mutation_method,
      child.get_binary()
    )
    gray = format_to_n_bits(
      binary_to_gray(binary),
      total_bits
    )
    valid_mutation = build_individual_if_valid(
      binary,
      gray,
      bits,
      sub_populations,
      precision,
      parsed_function,
      variables_array
    )

    if valid_mutation is not None:
      return valid_mutation

    if attempts > 5:
      return None

    attempts += 1

def mutate_children(
  children: List[Individual],
  mutation_method: str,
  mutation_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str],
  should_print: bool = False
) -> List[Individual]:
  if should_print:
    print(f'\nPopulation children before mutation: {children}\n')
    print()

  mutated_children: List[Individual] = []

  for child in children:
    mutated_child = child

    if random() < mutation_rate:
      if should_print:
        print(f'  Mutation for child: {child}\n')

      valid_mutation = mutate_individual(
        child,
        mutation_method,
        sub_populations,
        precision,
        parsed_function,
        variables_array
      )

      if valid_mutation is not None:
        mutated_child = valid_mutation

        if should_print:
          print(f'  Mutation for child completed: {mutated_child}\n')

    mutated_children.append(mutated_child)

  if should_print:
    print(f'\nPopulation children after mutation: {mutated_children}\n')
    print()

  return mutated_children
