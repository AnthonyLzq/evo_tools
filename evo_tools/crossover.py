from random import choice, random, sample
from typing import List, Tuple, Union

from evo_tools.bin_gray import generate_random_binary_with_a_len
from evo_tools.models import Individual, SubPopulation
from evo_tools.phenotype import build_individual_if_valid
from evo_tools.selection import select_parents


CROSSOVER_METHODS = (
  'one_point',
  'two_points',
  'uniform'
)

def _build_valid_children(
  binary_children: List[str],
  gray_children: List[str],
  bits: Tuple[int, ...],
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> List[Individual]:
  valid_children: List[Individual] = []

  for binary_child, gray_child in zip(binary_children, gray_children):
    valid_child = build_individual_if_valid(
      binary_child,
      gray_child,
      bits,
      sub_populations,
      precision,
      parsed_function,
      variables_array
    )

    if valid_child is not None:
      valid_children.append(valid_child)

  return valid_children

def crossover_one_point(
  parents: List[Tuple[Individual, Individual]],
  crossover_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> List[Individual]:
  if len(parents) == 0:
    return []

  total_bits = parents[0][0].get_total_bits()
  bits = parents[0][0].get_bits_tuple()
  children: List[Individual] = []

  for parent in parents:
    if random() < crossover_rate:
      points = [i for i in range(0, total_bits)]
      first_parent, second_parent = parent
      binary_p1, binary_p2 = first_parent.get_binary(), second_parent.get_binary()
      gray_p1, gray_p2 = first_parent.get_gray(), second_parent.get_gray()
      binary_children: List[str] = []
      gray_children: List[str] = []
      attempts = 0
      valid_children: List[Individual] = []

      while True:
        point = choice(points)
        binary_children = [
          binary_p1[:point] + binary_p2[point:],
          binary_p2[:point] + binary_p1[point:]
        ]
        gray_children = [
          gray_p1[:point] + gray_p2[point:],
          gray_p2[:point] + gray_p1[point:]
        ]
        valid_children = _build_valid_children(
          binary_children,
          gray_children,
          bits,
          sub_populations,
          precision,
          parsed_function,
          variables_array
        )

        if len(valid_children) > 0 or attempts >= total_bits:
          break

        points = [candidate for candidate in points if candidate != point]

        if len(points) == 0:
          break

        attempts += 1

      children.extend(valid_children)

  return children

def crossover_two_points(
  parents: List[Tuple[Individual, Individual]],
  crossover_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> List[Individual]:
  if len(parents) == 0:
    return []

  total_bits = parents[0][0].get_total_bits()
  bits = parents[0][0].get_bits_tuple()
  children: List[Individual] = []

  for parent in parents:
    if random() < crossover_rate:
      points = [i for i in range(0, total_bits)]
      first_parent, second_parent = parent
      binary_p1, binary_p2 = first_parent.get_binary(), second_parent.get_binary()
      gray_p1, gray_p2 = first_parent.get_gray(), second_parent.get_gray()
      binary_children: List[str] = []
      gray_children: List[str] = []
      attempts = 0
      valid_children: List[Individual] = []

      while True:
        point_1, point_2 = sample(points, 2)
        point_max, point_min = point_1, point_2

        if point_max < point_2:
          point_max = point_2

        if point_min > point_1:
          point_min = point_1

        binary_children = [
          binary_p1[:point_min] + binary_p2[point_min:point_max] + binary_p1[point_max:],
          binary_p2[:point_min] + binary_p1[point_min:point_max] + binary_p2[point_max:]
        ]
        gray_children = [
          gray_p1[:point_min] + gray_p2[point_min:point_max] + gray_p1[point_max:],
          gray_p2[:point_min] + gray_p1[point_min:point_max] + gray_p2[point_max:]
        ]
        valid_children = _build_valid_children(
          binary_children,
          gray_children,
          bits,
          sub_populations,
          precision,
          parsed_function,
          variables_array
        )

        if len(valid_children) > 0 or attempts >= total_bits:
          break

        attempts += 1

      children.extend(valid_children)

  return children

def crossover_uniform(
  parents: List[Tuple[Individual, Individual]],
  crossover_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> List[Individual]:
  if len(parents) == 0:
    return []

  total_bits = parents[0][0].get_total_bits()
  bits = parents[0][0].get_bits_tuple()
  children: List[Individual] = []

  for parent in parents:
    if random() < crossover_rate:
      first_parent, second_parent = parent
      binary_p1, binary_p2 = first_parent.get_binary(), second_parent.get_binary()
      gray_p1, gray_p2 = first_parent.get_gray(), second_parent.get_gray()
      mask = generate_random_binary_with_a_len(total_bits)
      binary_children: List[str] = ['', '']
      gray_children: List[str] = ['', '']

      for i, mask_element in enumerate(mask):
        if mask_element == '0':
          binary_children[0] += binary_p1[i]
          gray_children[0] += gray_p1[i]
          binary_children[1] += binary_p2[i]
          gray_children[1] += gray_p2[i]
        else:
          binary_children[0] += binary_p2[i]
          gray_children[0] += gray_p2[i]
          binary_children[1] += binary_p1[i]
          gray_children[1] += gray_p1[i]

      children.extend(_build_valid_children(
        binary_children,
        gray_children,
        bits,
        sub_populations,
        precision,
        parsed_function,
        variables_array
      ))

  return children

def validate_crossover_method(crossover_method: str) -> None:
  if crossover_method in CROSSOVER_METHODS:
    return

  raise Exception('Crossover method not allowed')

def apply_crossover(
  parents: List[Tuple[Individual, Individual]],
  crossover_method: str,
  crossover_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> List[Individual]:
  crossover_functions = {
    CROSSOVER_METHODS[0]: crossover_one_point,
    CROSSOVER_METHODS[1]: crossover_two_points,
    CROSSOVER_METHODS[2]: crossover_uniform
  }

  try:
    return crossover_functions[crossover_method](
      parents,
      crossover_rate,
      sub_populations,
      precision,
      parsed_function,
      variables_array
    )
  except KeyError as exc:
    raise Exception('Crossover method not allowed') from exc

def generate_children(
  population: List[Individual],
  seed: float,
  crossover_method: str,
  parent_selection_method: str,
  minimize: bool,
  crossover_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str]
) -> List[Individual]:
  parents = select_parents(
    population,
    seed,
    parent_selection_method,
    minimize
  )

  if len(parents) == 0:
    return []

  return apply_crossover(
    parents,
    crossover_method,
    crossover_rate,
    sub_populations,
    precision,
    parsed_function,
    variables_array
  )
