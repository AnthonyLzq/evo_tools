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

def _unique_parent_indexes(
  random_parents_indexes_chosen: np.ndarray
) -> List[Tuple[int, int]]:
  return [
    (i_1, i_2)
    for i_1, i_2 in np.unique(
      random_parents_indexes_chosen,
      axis = 0
    ).tolist()
    if i_1 != i_2
  ]

def _build_parent_pairs(
  population: List[Individual],
  parent_indexes: List[Tuple[int, int]]
) -> List[Tuple[Individual, Individual]]:
  return [
    (population[i_1], population[i_2])
    for i_1, i_2 in parent_indexes
  ]

def _select_parents_by_probabilities(
  population: List[Individual],
  seed: float,
  probabilities: np.ndarray
) -> List[Tuple[Individual, Individual]]:
  random_parents_indexes_chosen = np.random.choice(
    len(population),
    size = (round(seed), 2),
    p = probabilities
  )

  return _build_parent_pairs(
    population,
    _unique_parent_indexes(random_parents_indexes_chosen)
  )

def _generation_scores(population: List[Individual]) -> np.ndarray:
  return np.fromiter(
    (individual.get_score() for individual in population),
    dtype = float
  )

def select_parents_by_fitness_proportionate(
  population: List[Individual],
  seed: float
) -> List[Tuple[Individual, Individual]]:
  generation_scores = _generation_scores(population)
  generation_score = float(generation_scores.sum())
  shifted_generation_scores = generation_scores + generation_score / 2

  return _select_parents_by_probabilities(
    population,
    seed,
    shifted_generation_scores / (
      generation_score + generation_score / 2 * len(generation_scores)
    )
  )

def select_parents_by_roulette(
  population: List[Individual],
  seed: float
) -> List[Tuple[Individual, Individual]]:
  generation_scores = _generation_scores(population)
  generation_score = float(generation_scores.sum())

  return _select_parents_by_probabilities(
    population,
    seed,
    generation_scores / generation_score
  )

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
