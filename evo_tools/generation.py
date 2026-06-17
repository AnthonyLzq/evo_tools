from typing import List, Tuple, Union

from evo_tools.crossover import generate_children
from evo_tools.initialization import select_initial_population
from evo_tools.models import Individual, SubPopulation
from evo_tools.mutation import mutate_children
from evo_tools.scoring import population_fitness_average, population_score_stats, \
  rank_population, selection_strength


def _compose_next_population(
  current_population: List[Individual],
  mutated_individuals: List[Individual],
  sample_size: int
) -> List[Individual]:
  current_survivor_count = max(sample_size - len(mutated_individuals), 0)
  current_survivors = current_population[:current_survivor_count]
  remaining_slots = sample_size - len(current_survivors)

  return current_survivors + mutated_individuals[:remaining_slots]

def select_next_generation(
  current_population: List[Individual],
  individuals: List[Individual],
  minimize: bool,
  sample_size: int,
  mutation_method: str,
  mutation_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str],
  objective_function,
  should_print: bool = False
) -> Tuple[List[Individual], float]:
  mutated_individuals = mutate_children(
    individuals,
    mutation_method,
    mutation_rate,
    sub_populations,
    precision,
    parsed_function,
    variables_array,
    should_print
  )
  rank_population(
    mutated_individuals,
    minimize,
    sub_populations,
    precision,
    objective_function,
    should_print
  )

  score_mean_before_selection, score_std_before_selection = population_score_stats(
    current_population
  )
  next_population = _compose_next_population(
    current_population,
    mutated_individuals,
    sample_size
  )
  rank_population(
    next_population,
    minimize,
    sub_populations,
    precision,
    objective_function,
    should_print
  )

  return next_population, selection_strength(
    next_population,
    score_mean_before_selection,
    score_std_before_selection
  )

def best_individual(current_population: List[Individual]) -> Individual:
  return current_population[0]

def initialize_canonical_state(
  initial_population: List[Individual],
  sample_size: int,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str],
  objective_function,
  minimize: bool,
  should_print: bool = False
) -> Tuple[List[Individual], List[Individual], Individual, int, List[float], List[float]]:
  initial_population, current_population = select_initial_population(
    initial_population,
    sample_size,
    sub_populations,
    precision,
    parsed_function,
    variables_array,
    should_print
  )
  rank_population(
    current_population,
    minimize,
    sub_populations,
    precision,
    objective_function,
    should_print
  )

  return (
    initial_population,
    current_population,
    best_individual(current_population),
    1,
    [],
    [population_fitness_average(current_population)]
  )

def run_canonical_iteration(
  current_population: List[Individual],
  sample_size: int,
  seed: float,
  crossover_method,
  parent_selection_method,
  mutation_method: str,
  minimize: bool,
  crossover_rate: float,
  mutation_rate: float,
  sub_populations: List[SubPopulation],
  precision: Union[float, int],
  parsed_function,
  variables_array: List[str],
  objective_function,
  should_print: bool = False
) -> Tuple[List[Individual], Individual, float, float]:
  children = generate_children(
    current_population,
    sample_size * seed,
    crossover_method,
    parent_selection_method,
    minimize,
    crossover_rate,
    sub_populations,
    precision,
    parsed_function,
    variables_array
  )
  current_population, current_selection_strength = select_next_generation(
    current_population,
    children,
    minimize,
    sample_size,
    mutation_method,
    mutation_rate,
    sub_populations,
    precision,
    parsed_function,
    variables_array,
    objective_function,
    should_print
  )

  return (
    current_population,
    best_individual(current_population),
    current_selection_strength,
    population_fitness_average(current_population)
  )
