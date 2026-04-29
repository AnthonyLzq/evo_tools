from typing import List, Tuple, Union

from evo_tools.models import Individual, SubPopulation
from evo_tools.mutation import mutate_children
from evo_tools.scoring import population_score_stats, rank_population, \
  selection_strength


def _compose_next_population(
  current_population: List[Individual],
  mutated_individuals: List[Individual],
  sample_size: int
) -> List[Individual]:
  return current_population[
    :len(current_population) - len(mutated_individuals)
  ] + mutated_individuals[:sample_size]

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
    parsed_function,
    variables_array,
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
    parsed_function,
    variables_array,
    should_print
  )

  return next_population, selection_strength(
    next_population,
    score_mean_before_selection,
    score_std_before_selection
  )
