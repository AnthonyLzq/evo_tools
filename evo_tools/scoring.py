from typing import List, Tuple

import numpy as np

from evo_tools.models import Individual, SubPopulation
from evo_tools.phenotype import evaluate_individual_objective

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

def population_fitness_average(population_sample: List[Individual]) -> float:
  return float(np.mean(
    np.array(
      [individual.get_fitness() for individual in population_sample] # type: ignore
    )
  ))

def population_score_stats(
  population_sample: List[Individual]
) -> Tuple[float, float]:
  population_scores = np.array(
    [individual.get_score() for individual in population_sample]
  )

  return (
    float(np.mean(population_scores)),
    float(np.std(population_scores))
  )

def selection_strength(
  population_sample: List[Individual],
  score_mean_before_selection: float,
  score_std_before_selection: float
) -> float:
  score_mean_after_selection, _ = population_score_stats(population_sample)

  return (
    float(abs(
      (score_mean_after_selection - score_mean_before_selection) / score_std_before_selection
    )) if score_std_before_selection > 0 else 0.0
  )

def rank_population(
  population_sample: List[Individual],
  minimize: bool,
  sub_populations: List[SubPopulation],
  precision,
  parsed_function,
  variables_array: List[str],
  should_print: bool = False
) -> None:
  if len(population_sample) == 0:
    return

  objective_values: List[float] = []

  for i, individual in enumerate(population_sample):
    individual.set_score(0)
    individual.set_objective_value(None)
    objective_value = evaluate_individual_objective(
      individual,
      i,
      sub_populations,
      precision,
      parsed_function,
      variables_array,
      should_print
    )

    if objective_value is not None:
      objective_values.append(objective_value)
      individual.set_objective_value(objective_value)

  if len(objective_values) > 0:
    assign_scores(population_sample, objective_values, minimize)

  sort_population_by_score(population_sample)
