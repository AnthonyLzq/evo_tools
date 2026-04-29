from random import sample
from typing import List, Tuple

import numpy as np

from evo_tools.models import Individual

PARENT_SELECTION_METHODS = (
  'fitness_proportionate',
  'roulette',
  'tournament'
)

def validate_parent_selection_method(parent_selection_method: str) -> None:
  if parent_selection_method in PARENT_SELECTION_METHODS:
    return

  raise Exception('Parent selection method not allowed')

def select_parents(
  population: List[Individual],
  seed: float,
  parent_selection_method: str,
  minimize: bool
) -> List[Tuple[Individual, Individual]]:
  parent_selection_strategies = {
    PARENT_SELECTION_METHODS[0]: lambda: select_parents_by_fitness_proportionate(
      population,
      seed
    ),
    PARENT_SELECTION_METHODS[1]: lambda: select_parents_by_roulette(
      population,
      seed
    ),
    PARENT_SELECTION_METHODS[2]: lambda: select_parents_by_tournament(
      population,
      seed,
      10,
      minimize
    )
  }

  try:
    return parent_selection_strategies[parent_selection_method]()
  except KeyError as exc:
    raise Exception('Parent selection method not allowed') from exc

def select_parents_by_fitness_proportionate(
  population: List[Individual],
  seed: float
) -> List[Tuple[Individual, Individual]]:
  total_population = len(population)
  parents_candidates = np.array(population)
  generation_scores = np.array(
    [individual.get_score() for individual in parents_candidates]
  )
  generation_score = sum(generation_scores)
  random_parents_indexes_chosen = np.random.choice(
    total_population,
    size = (round(seed), 2),
    p = np.array(
      [x + generation_score / 2 for x in generation_scores]
    ) / (generation_score + generation_score / 2 * len(generation_scores))
  )
  unique_random_parents_indexes_chosen, _ = np.unique(
    [
      str(
        np.ndarray.tolist(index)
      )[1:-1].replace(' ', '') for index in random_parents_indexes_chosen
    ],
    return_index = True
  )
  final_parents_indexes = list(
    filter(
      lambda a: a[0] != a[1],
      map(
        lambda e: [int(i) for i in e.split(',')],
        unique_random_parents_indexes_chosen
      )
    )
  )
  parents: List[Tuple[Individual, Individual]] = []

  for indexes in final_parents_indexes:
    i_1, i_2 = indexes
    parents.append((
      population[i_1],
      population[i_2]
    ))

  return parents

def select_parents_by_roulette(
  population: List[Individual],
  seed: float
) -> List[Tuple[Individual, Individual]]:
  total_population = len(population)
  parents_candidates = np.array(population)
  generation_scores = np.array(
    [individual.get_score() for individual in parents_candidates]
  )
  generation_score = sum(generation_scores)
  random_parents_indexes_chosen = np.random.choice(
    total_population,
    size = (round(seed), 2),
    p = generation_scores / generation_score
  )
  unique_random_parents_indexes_chosen, _ = np.unique(
    [
      str(
        np.ndarray.tolist(index)
      )[1:-1].replace(' ', '') for index in random_parents_indexes_chosen
    ],
    return_index = True
  )
  final_parents_indexes = list(
    filter(
      lambda a: a[0] != a[1],
      map(
        lambda e: [int(i) for i in e.split(',')],
        unique_random_parents_indexes_chosen
      )
    )
  )
  parents: List[Tuple[Individual, Individual]] = []

  for indexes in final_parents_indexes:
    i_1, i_2 = indexes
    parents.append((
      population[i_1],
      population[i_2]
    ))

  return parents

def select_parents_by_tournament(
  population: List[Individual],
  seed: float,
  k: int,
  minimize: bool
) -> List[Tuple[Individual, Individual]]:
  parents: List[Tuple[Individual, Individual]] = []

  while len(parents) < seed:
    chosen_list: List[Individual] = []

    for _ in range(2):
      candidates = sample(population, min(k, len(population)))
      chosen = min(
        candidates,
        key = lambda individual: individual.get_fitness()
      ) if minimize else max(
        candidates,
        key = lambda individual: individual.get_fitness()
      )
      chosen_list.append(chosen)

    parents.append((chosen_list[0], chosen_list[1]))

  return parents
