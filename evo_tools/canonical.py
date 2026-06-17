from time import time
from typing import Dict, List, Tuple, Union

from sympy import exp

from evo_tools.crossover import validate_crossover_method
from evo_tools.generation import initialize_canonical_state, run_canonical_iteration
from evo_tools.models import Individual, SubPopulation
from evo_tools.mutation import validate_mutation_method
from evo_tools.phenotype import build_solution
from evo_tools.reporting import print_final_summary, print_iteration_summary
from evo_tools.selection import validate_parent_selection_method

DIVERSITY_METRICS = (
  'genotype_unique_ratio',
)

def validate_canonical_methods(
  parent_selection_method: str,
  crossover_method: str,
  mutation_method: str
) -> None:
  validate_parent_selection_method(parent_selection_method)
  validate_crossover_method(crossover_method)
  validate_mutation_method(mutation_method)

def validate_early_stopping_parameters(
  early_stopping: bool,
  min_iterations: int,
  patience: int,
  tolerance: float,
  diversity_early_stopping: bool = False,
  diversity_metric: str = 'genotype_unique_ratio',
  diversity_threshold: float = 0.05
) -> None:
  if not isinstance(early_stopping, bool):
    raise ValueError('EARLY_STOPPING must be a boolean')

  if not isinstance(diversity_early_stopping, bool):
    raise ValueError('DIVERSITY_EARLY_STOPPING must be a boolean')

  if diversity_early_stopping and not early_stopping:
    raise ValueError('DIVERSITY_EARLY_STOPPING requires EARLY_STOPPING')

  if not early_stopping:
    return

  if min_iterations < 1:
    raise ValueError('EARLY_STOPPING_MIN_ITERATIONS must be at least 1')

  if patience < 1:
    raise ValueError('EARLY_STOPPING_PATIENCE must be at least 1')

  if tolerance < 0:
    raise ValueError('EARLY_STOPPING_TOLERANCE must be non-negative')

  if not diversity_early_stopping:
    return

  if diversity_metric not in DIVERSITY_METRICS:
    raise ValueError('DIVERSITY_METRIC not allowed')

  if diversity_threshold < 0 or diversity_threshold > 1:
    raise ValueError('DIVERSITY_THRESHOLD must be between 0 and 1')

def _genotype_unique_ratio(population: List[Individual]) -> float:
  if len(population) == 0:
    return 0.0

  return len({
    individual.get_binary()
    for individual in population
  }) / len(population)

def _population_diversity_ratio(
  population: List[Individual],
  diversity_metric: str
) -> float:
  if diversity_metric == DIVERSITY_METRICS[0]:
    return _genotype_unique_ratio(population)

  raise ValueError('DIVERSITY_METRIC not allowed')

def _has_best_objective_improved(
  current_objective: float,
  best_objective: float,
  minimize: bool,
  tolerance: float
) -> bool:
  if minimize:
    return current_objective < best_objective - tolerance

  return current_objective > best_objective + tolerance

def _update_early_stopping_state(
  current_objective: float,
  best_objective: float,
  minimize: bool,
  tolerance: float,
  stalled_iterations: int
) -> Tuple[float, int]:
  if _has_best_objective_improved(
    current_objective,
    best_objective,
    minimize,
    tolerance
  ):
    return current_objective, 0

  return best_objective, stalled_iterations + 1

def _should_stop_early(
  early_stopping: bool,
  current_iteration: int,
  min_iterations: int,
  stalled_iterations: int,
  patience: int,
  diversity_early_stopping: bool = False,
  diversity_ratio: float = 0.0,
  diversity_threshold: float = 0.0
) -> bool:
  should_stop_by_progress = (
    early_stopping and
    current_iteration >= min_iterations and
    stalled_iterations >= patience
  )

  if not should_stop_by_progress:
    return False

  if not diversity_early_stopping:
    return True

  return diversity_ratio <= diversity_threshold

def finalize_canonical_result(
  best_individual: Individual,
  parsed_function,
  variables_array: List[str],
  current_iteration: int,
  selection_strength: float,
  fitness_avg_list: List[float],
  should_print: bool = False
) -> Tuple[Dict[str, float], exp]:
  solution, function = build_solution(
    best_individual.get_numbers_tuple(),
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
  mutation_rate: float,
  early_stopping: bool,
  early_stopping_min_iterations: int,
  early_stopping_patience: int,
  early_stopping_tolerance: float,
  diversity_early_stopping: bool,
  diversity_metric: str,
  diversity_threshold: float
) -> Tuple[List[Individual], List[Individual], Individual, float, int, List[float], List[float]]:
  start = time()
  initial_population, current_population, best_individual, current_iteration, \
    scores, fitness_avg_list = initialize_canonical_state(
      initial_population,
      sample_size,
      sub_populations,
      precision,
      parsed_function,
      variables_array,
      objective_function,
      minimize,
      should_print
    )
  selection_strength = 0.0
  best_objective = best_individual.get_fitness()
  stalled_iterations = 0
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
    best_objective, stalled_iterations = _update_early_stopping_state(
      best_individual.get_fitness(),
      best_objective,
      minimize,
      early_stopping_tolerance,
      stalled_iterations
    )
    diversity_ratio = (
      _population_diversity_ratio(current_population, diversity_metric)
      if diversity_early_stopping else
      0.0
    )
    end = time()

    if _should_stop_early(
      early_stopping,
      current_iteration,
      early_stopping_min_iterations,
      stalled_iterations,
      early_stopping_patience,
      diversity_early_stopping,
      diversity_ratio,
      diversity_threshold
    ):
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
