from random import choice, random, sample
from typing import List, Tuple, Union

from sympy import sympify

from evo_tools.bin_gray import generate_random_binary_with_a_len
from evo_tools.helpers import sub_strings_by_array
from evo_tools.models import Individual, SubPopulation
from evo_tools.phenotype import chromosome_to_numbers_repr, validate_binaries_in_range


def _build_child(
  binary: str,
  gray: str,
  bits: List[int],
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  function,
  variables: str
) -> Individual:
  return Individual(
    binary,
    gray,
    0,
    bits,
    chromosome_to_numbers_repr(binary, bits, sub_populations, precision),
    sympify(str(function)),
    variables.split()
  )


def _validate_children(
  binary_children: List[str],
  bits: List[int],
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> Tuple[bool, bool]:
  return (
    validate_binaries_in_range(
      [sub_strings_by_array(binary_children[0], bits)],
      sub_populations,
      precision
    ),
    validate_binaries_in_range(
      [sub_strings_by_array(binary_children[1], bits)],
      sub_populations,
      precision
    )
  )


def _append_valid_children(
  children: List[Individual],
  binary_children: List[str],
  gray_children: List[str],
  bits: List[int],
  first_child_is_valid: bool,
  second_child_is_valid: bool,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  function,
  variables: str
) -> None:
  if first_child_is_valid:
    children.append(
      _build_child(
        binary_children[0],
        gray_children[0],
        bits,
        sub_populations,
        precision,
        function,
        variables
      )
    )

  if second_child_is_valid:
    children.append(
      _build_child(
        binary_children[1],
        gray_children[1],
        bits,
        sub_populations,
        precision,
        function,
        variables
      )
    )


def crossover_one_point(
  parents: List[Tuple[Individual, Individual]],
  crossover_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  function,
  variables: str
) -> List[Individual]:
  if len(parents) == 0:
    return []

  total_bits = parents[0][0].get_total_bits()
  bits = parents[0][0].get_bits()
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
      first_child_is_valid = False
      second_child_is_valid = False

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
        first_child_is_valid, second_child_is_valid = _validate_children(
          binary_children,
          bits,
          sub_populations,
          precision
        )

        if first_child_is_valid or second_child_is_valid or attempts >= total_bits:
          break

        points = [candidate for candidate in points if candidate != point]

        if len(points) == 0:
          break

        attempts += 1

      _append_valid_children(
        children,
        binary_children,
        gray_children,
        bits,
        first_child_is_valid,
        second_child_is_valid,
        sub_populations,
        precision,
        function,
        variables
      )

  return children


def crossover_two_points(
  parents: List[Tuple[Individual, Individual]],
  crossover_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  function,
  variables: str
) -> List[Individual]:
  if len(parents) == 0:
    return []

  total_bits = parents[0][0].get_total_bits()
  bits = parents[0][0].get_bits()
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
      first_child_is_valid = False
      second_child_is_valid = False

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
        first_child_is_valid, second_child_is_valid = _validate_children(
          binary_children,
          bits,
          sub_populations,
          precision
        )

        if first_child_is_valid or second_child_is_valid or attempts >= total_bits:
          break

        attempts += 1

      _append_valid_children(
        children,
        binary_children,
        gray_children,
        bits,
        first_child_is_valid,
        second_child_is_valid,
        sub_populations,
        precision,
        function,
        variables
      )

  return children


def crossover_uniform(
  parents: List[Tuple[Individual, Individual]],
  crossover_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  function,
  variables: str
) -> List[Individual]:
  if len(parents) == 0:
    return []

  total_bits = parents[0][0].get_total_bits()
  bits = parents[0][0].get_bits()
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

      first_child_is_valid, second_child_is_valid = _validate_children(
        binary_children,
        bits,
        sub_populations,
        precision
      )
      _append_valid_children(
        children,
        binary_children,
        gray_children,
        bits,
        first_child_is_valid,
        second_child_is_valid,
        sub_populations,
        precision,
        function,
        variables
      )

  return children
