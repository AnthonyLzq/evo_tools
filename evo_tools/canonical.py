from typing import Dict, List, Tuple, Union

from sympy import exp

from evo_tools.crossover import validate_crossover_method
from evo_tools.models import Individual, SubPopulation
from evo_tools.mutation import validate_mutation_method
from evo_tools.phenotype import build_solution, decode_individual
from evo_tools.reporting import print_final_summary
from evo_tools.selection import validate_parent_selection_method


def validate_canonical_methods(
  parent_selection_method: str,
  crossover_method: str,
  mutation_method: str
) -> None:
  validate_parent_selection_method(parent_selection_method)
  validate_crossover_method(crossover_method)
  validate_mutation_method(mutation_method)

def finalize_canonical_result(
  best_individual: Individual,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str],
  current_iteration: int,
  selection_strength: float,
  fitness_avg_list: List[float],
  should_print: bool = False
) -> Tuple[Dict[str, float], exp]:
  _, floats = decode_individual(
    best_individual,
    sub_populations,
    precision
  )
  solution, function = build_solution(
    floats,
    parsed_function,
    variables_array
  )

  if should_print:
    print_final_summary(
      current_iteration,
      best_individual,
      selection_strength,
      fitness_avg_list,
      solution,
      function
    )

  return solution, function
