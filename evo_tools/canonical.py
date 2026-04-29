from time import time
from typing import Dict, List, Tuple, Union

from sympy import exp

from evo_tools.crossover import validate_crossover_method
from evo_tools.generation import initialize_canonical_state, run_canonical_iteration
from evo_tools.models import Individual, SubPopulation
from evo_tools.mutation import validate_mutation_method
from evo_tools.phenotype import build_solution, decode_individual
from evo_tools.reporting import print_final_summary, print_iteration_summary
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

def run_canonical_algorithm(
  initial_population: List[Individual],
  sample_size: int,
  max_sample_size: int,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str],
  objective_function,
  minimize: bool,
  iterations: int,
  seed: float,
  should_print: bool,
  parent_selection_method: str,
  crossover_method: str,
  mutation_method: str,
  crossover_rate: float,
  mutation_rate: float
) -> Tuple[List[Individual], List[Individual], Individual, float, int, List[float], List[float]]:
  start = time()
  initial_population, current_population, best_individual, current_iteration, \
    scores, fitness_avg_list = initialize_canonical_state(
      initial_population,
      sample_size,
      max_sample_size,
      sub_populations,
      precision,
      parsed_function,
      variables_array,
      objective_function,
      minimize,
      should_print
    )
  selection_strength = 0.0
  end = time()

  if should_print:
    print_iteration_summary(
      current_iteration,
      best_individual,
      selection_strength,
      end - start,
      current_population
    )

  for _ in range(iterations - 1):
    start = time()
    current_iteration += 1
    current_population, best_individual, selection_strength, fitness_avg = \
      run_canonical_iteration(
        current_population,
        sample_size,
        seed,
        crossover_method,
        parent_selection_method,
        mutation_method,
        minimize,
        crossover_rate,
        mutation_rate,
        sub_populations,
        precision,
        parsed_function,
        variables_array,
        objective_function,
        should_print
      )
    scores.append(best_individual.get_score())
    fitness_avg_list.append(fitness_avg)
    end = time()

    if selection_strength <= 1e-4:
      break

    if should_print:
      print_iteration_summary(
        current_iteration,
        best_individual,
        selection_strength,
        end - start,
        current_population
      )

  return (
    initial_population,
    current_population,
    best_individual,
    selection_strength,
    current_iteration,
    scores,
    fitness_avg_list
  )
