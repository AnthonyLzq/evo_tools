from typing import List, Sequence, TypeVar, Union

import numpy as np

T = TypeVar('T')
_RNG = np.random.default_rng()


def seed_random_generators(random_seed: Union[int, None]) -> None:
  global _RNG

  if random_seed is None:
    return

  _RNG = np.random.default_rng(random_seed)

def random_probability() -> float:
  return float(_RNG.random())

def randint_inclusive(lower_bound: int, upper_bound: int) -> int:
  return int(_RNG.integers(lower_bound, upper_bound + 1))

def choice_from(items: Sequence[T]) -> T:
  return items[randint_inclusive(0, len(items) - 1)]

def sample_without_replacement(items: Sequence[T], size: int) -> List[T]:
  if size == 0:
    return []

  indexes = _RNG.choice(len(items), size = size, replace = False)

  return [items[int(index)] for index in np.atleast_1d(indexes)]

def sample_with_replacement(items: Sequence[T], size: int) -> List[T]:
  if size == 0:
    return []

  indexes = _RNG.choice(len(items), size = size, replace = True)

  return [items[int(index)] for index in np.atleast_1d(indexes)]

def sample_index_pairs_by_probabilities(
  population_size: int,
  pair_count: int,
  probabilities: np.ndarray
) -> np.ndarray:
  if pair_count == 0:
    return np.empty((0, 2), dtype = int)

  return _RNG.choice(
    population_size,
    size = (pair_count, 2),
    p = probabilities
  )
