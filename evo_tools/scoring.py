from typing import List

from evo_tools.models import Individual

def assign_scores(
  population_sample: List[Individual],
  objective_values: List[float],
  minimize: bool
) -> None:
  if len(objective_values) == 0:
    return

  valid_population_sample = [
    individual for individual in population_sample if individual._objective_value is not None
  ]
  reference_value = max(objective_values) if minimize else min(objective_values)

  for i, objective_value in enumerate(objective_values):
    score = (
      1e-3 + reference_value - objective_value
      if minimize else
      1e-3 + objective_value - reference_value
    )
    valid_population_sample[i].set_score(score)

def sort_population_by_score(population_sample: List[Individual]) -> None:
  population_sample.sort(reverse = True, key = lambda x: x.get_score())
