import numpy as np
from sympy import sympify
from unittest.mock import patch

from evo_tools.generation import initialize_canonical_state
from evo_tools.population import Population
from evo_tools.scoring import rank_population
from evo_tools.selection import select_parents_by_fitness_proportionate, \
  select_parents_by_roulette, select_parents_by_tournament

def build_population(rng, precision, function = 'x', sample_size = None):
  domain_size = int(abs(rng[1] - rng[0]) / precision) + 1
  population = Population(
    [rng],
    precision,
    1,
    0.01,
    'x',
    sympify(function),
    sample_size = sample_size or domain_size
  )
  population._initial_population, population._current_population, \
    population._best_individual, _, _, _ = initialize_canonical_state(
      population._initial_population,
    population._sample_size,
    population._max_sample_size,
    population._sub_populations,
    population._precision,
    population._parsed_function,
    population._variables_array,
    population._objective_function,
    True
  )

  return population

def rank_current_population(population, minimize):
  rank_population(
    population._current_population.copy(),
    minimize,
    population._sub_populations,
    population._precision,
    population._objective_function
  )

def test_fitness_scores_follow_optimization_direction() -> None:
  population = build_population((0, 2), 1)
  current_population = population._current_population.copy()

  rank_current_population(population, True)
  minimize_scores = {
    individual.get_fitness(): individual.get_score()
    for individual in current_population
  }

  rank_current_population(population, False)
  maximize_scores = {
    individual.get_fitness(): individual.get_score()
    for individual in current_population
  }

  assert minimize_scores[0.0] > minimize_scores[1.0] > minimize_scores[2.0]
  assert maximize_scores[2.0] > maximize_scores[1.0] > maximize_scores[0.0]

def test_roulette_selection_probabilities_follow_maximization_scores() -> None:
  population = build_population((0, 2), 1)
  current_population = population._current_population.copy()
  rank_current_population(population, False)
  captured = {}

  def fake_choice(total_population, size, p):
    captured['p'] = p.tolist()

    return np.array([[0, 1]])

  with patch('evo_tools.selection.np.random.choice', side_effect = fake_choice):
    select_parents_by_roulette(population._current_population, 1)

  probabilities_by_objective = sorted(
    (individual.get_fitness(), captured['p'][index])
    for index, individual in enumerate(population._current_population)
  )

  assert probabilities_by_objective[0][1] < probabilities_by_objective[1][1] < probabilities_by_objective[2][1]

def test_fitness_proportionate_probabilities_follow_maximization_scores() -> None:
  population = build_population((0, 2), 1)
  current_population = population._current_population.copy()
  rank_current_population(population, False)
  captured = {}

  def fake_choice(total_population, size, p):
    captured['p'] = p.tolist()

    return np.array([[0, 1]])

  with patch('evo_tools.selection.np.random.choice', side_effect = fake_choice):
    select_parents_by_fitness_proportionate(population._current_population, 1)

  probabilities_by_objective = sorted(
    (individual.get_fitness(), captured['p'][index])
    for index, individual in enumerate(population._current_population)
  )

  assert probabilities_by_objective[0][1] < probabilities_by_objective[1][1] < probabilities_by_objective[2][1]

def test_tournament_selection_uses_raw_objective_values() -> None:
  population = build_population((-2, 1), 1)
  current_population = population._current_population.copy()
  rank_current_population(population, False)

  maximize_parents = select_parents_by_tournament(
    population._current_population,
    1,
    len(current_population),
    False
  )
  minimize_parents = select_parents_by_tournament(
    population._current_population,
    1,
    len(current_population),
    True
  )

  assert all(parent.get_fitness() == 1.0 for pair in maximize_parents for parent in pair)
  assert all(parent.get_fitness() == -2.0 for pair in minimize_parents for parent in pair)
