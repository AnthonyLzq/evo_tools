import numpy as np
import pytest
from sympy import Piecewise, symbols, sympify
from unittest.mock import patch

from evo_tools.generation import _compose_next_population, initialize_canonical_state
from evo_tools.initialization import select_initial_population
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

def test_rank_population_reuses_cached_objective_values() -> None:
  population = build_population((0, 2), 1)
  rank_population(
    population._current_population,
    False,
    population._sub_populations,
    population._precision,
    population._objective_function
  )

  with patch(
    'evo_tools.scoring.evaluate_individual_objective',
    side_effect = AssertionError('unexpected reevaluation')
  ):
    rank_population(
      population._current_population,
      False,
      population._sub_populations,
      population._precision,
      population._objective_function
    )

  assert all(individual.has_objective_value() for individual in population._current_population)

def test_select_initial_population_keeps_distinct_population_lists() -> None:
  population = Population(
    [(0, 2)],
    1,
    1,
    0.01,
    'x',
    sympify('x')
  )
  initial_population, current_population = select_initial_population(
    [],
    population._sample_size,
    population._sub_populations,
    population._precision,
    population._parsed_function,
    population._variables_array
  )

  assert initial_population is not current_population
  assert initial_population == current_population

def test_select_initial_population_generated_lists_do_not_share_container() -> None:
  population = Population(
    [(0, 2)],
    1,
    1,
    0.01,
    'x',
    sympify('x')
  )
  initial_population, current_population = select_initial_population(
    [],
    population._sample_size,
    population._sub_populations,
    population._precision,
    population._parsed_function,
    population._variables_array
  )

  current_population.pop()

  assert len(initial_population) == population._sample_size
  assert len(current_population) == population._sample_size - 1

def test_compose_next_population_keeps_sample_size_with_large_offspring_batch() -> None:
  population = build_population((0, 9), 1)
  current_population = population._current_population[:5]
  mutated_individuals = population._current_population[5:] + population._current_population

  next_population = _compose_next_population(
    current_population,
    mutated_individuals,
    len(current_population)
  )

  assert len(next_population) == len(current_population)
  assert next_population == mutated_individuals[:len(current_population)]

def test_select_initial_population_reuses_existing_initial_population_list() -> None:
  population = build_population((0, 2), 1)
  initial_population, current_population = select_initial_population(
    population._initial_population,
    population._sample_size,
    population._sub_populations,
    population._precision,
    population._parsed_function,
    population._variables_array
  )

  assert initial_population is population._initial_population
  assert current_population is not population._initial_population
  assert initial_population == current_population

def test_select_initial_population_existing_lists_do_not_share_container() -> None:
  population = build_population((0, 2), 1)
  initial_population, current_population = select_initial_population(
    population._initial_population,
    population._sample_size,
    population._sub_populations,
    population._precision,
    population._parsed_function,
    population._variables_array
  )

  current_population.pop()

  assert len(initial_population) == population._sample_size
  assert len(current_population) == population._sample_size - 1

def test_roulette_selection_probabilities_follow_maximization_scores() -> None:
  population = build_population((0, 2), 1)
  current_population = population._current_population.copy()
  rank_current_population(population, False)
  captured = {}

  def fake_choice(population_size, pair_count, probabilities):
    captured['p'] = probabilities.tolist()

    return np.array([[0, 1]])

  with patch(
    'evo_tools.selection.sample_index_pairs_by_probabilities',
    side_effect = fake_choice
  ):
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

  def fake_choice(population_size, pair_count, probabilities):
    captured['p'] = probabilities.tolist()

    return np.array([[0, 1]])

  with patch(
    'evo_tools.selection.sample_index_pairs_by_probabilities',
    side_effect = fake_choice
  ):
    select_parents_by_fitness_proportionate(population._current_population, 1)

  probabilities_by_objective = sorted(
    (individual.get_fitness(), captured['p'][index])
    for index, individual in enumerate(population._current_population)
  )

  assert probabilities_by_objective[0][1] < probabilities_by_objective[1][1] < probabilities_by_objective[2][1]

def test_roulette_selection_drops_duplicate_and_self_pairs() -> None:
  population = build_population((0, 11), 1)
  current_population = population._current_population.copy()
  rank_current_population(population, False)
  expected_pairs = {
    (
      current_population[10].get_fitness(),
      current_population[11].get_fitness()
    ),
    (
      current_population[2].get_fitness(),
      current_population[10].get_fitness()
    )
  }

  with patch(
    'evo_tools.selection.sample_index_pairs_by_probabilities',
    return_value = np.array([[10, 11], [10, 11], [11, 11], [2, 10]])
  ):
    parents = select_parents_by_roulette(population._current_population, 4)

  assert {
    (first.get_fitness(), second.get_fitness())
    for first, second in parents
  } == expected_pairs

def test_roulette_selection_uses_uniform_probabilities_when_scores_are_zero() -> None:
  population = build_population((0, 2), 1)
  captured = {}

  for individual in population._current_population:
    individual.set_score(0)

  def fake_choice(population_size, pair_count, probabilities):
    captured['p'] = probabilities.tolist()

    return np.array([[0, 1]])

  with patch(
    'evo_tools.selection.sample_index_pairs_by_probabilities',
    side_effect = fake_choice
  ):
    select_parents_by_roulette(population._current_population, 1)

  assert captured['p'] == [1 / 3, 1 / 3, 1 / 3]

def test_fitness_proportionate_selection_drops_duplicate_and_self_pairs() -> None:
  population = build_population((0, 11), 1)
  current_population = population._current_population.copy()
  rank_current_population(population, False)
  expected_pairs = {
    (
      current_population[10].get_fitness(),
      current_population[11].get_fitness()
    ),
    (
      current_population[2].get_fitness(),
      current_population[10].get_fitness()
    )
  }

  with patch(
    'evo_tools.selection.sample_index_pairs_by_probabilities',
    return_value = np.array([[10, 11], [10, 11], [11, 11], [2, 10]])
  ):
    parents = select_parents_by_fitness_proportionate(
      population._current_population,
      4
    )

  assert {
    (first.get_fitness(), second.get_fitness())
    for first, second in parents
  } == expected_pairs

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

def test_binary_population_can_exceed_single_variable_domain_size() -> None:
  population = Population(
    [(0, 1)] * 10,
    1,
    1,
    0.01,
    'x1 x2 x3 x4 x5 x6 x7 x8 x9 x10',
    sympify('x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10'),
    sample_size = 20
  )

  scores, solution, result, fitness_avg = population.canonical_algorithm(
    ITERATIONS = 3,
    MINIMIZE = False,
    PARENT_SELECTION_METHOD = 'tournament'
  )

  assert len(population._current_population) == 20
  assert len(scores) > 0
  assert len(fitness_avg) > 0
  assert float(result) >= 0
  assert all(value in (0.0, 1.0) for value in solution.values())

def test_population_sympifies_expression_without_string_roundtrip() -> None:
  expression = sympify('x + 1')
  captured = {}

  def fake_sympify(value, strict = False):
    captured['value'] = value
    captured['strict'] = strict

    return value

  with patch('evo_tools.population.sympify', side_effect = fake_sympify):
    Population(
      [(0, 2)],
      1,
      1,
      0.01,
      'x',
      expression
    )

  assert captured['value'] is expression
  assert captured['strict'] is True

def test_population_rejects_string_objective_functions() -> None:
  with pytest.raises(ValueError, match = 'SymPy expression'):
    Population(
      [(0, 2)],
      1,
      1,
      0.01,
      'x',
      'x + 1'
    )

def test_canonical_algorithm_solves_reference_knapsack_case() -> None:
  weights = [10, 20, 30, 5, 15, 25, 7, 12, 18, 3]
  values = [60, 100, 120, 30, 80, 90, 40, 70, 85, 20]
  capacity = 50
  variables = symbols('x1:11')
  total_weight = sum(
    weight * variable for weight, variable in zip(weights, variables)
  )
  total_value = sum(
    value * variable for value, variable in zip(values, variables)
  )
  objective = total_value - Piecewise(
    (0, total_weight <= capacity),
    (1000 * (total_weight - capacity), True)
  )

  population = Population(
    [(0, 1)] * len(variables),
    1,
    1,
    0.05,
    ' '.join(str(variable) for variable in variables),
    objective,
    sample_size = 60
  )

  _, solution, result, _ = population.canonical_algorithm(
    ITERATIONS = 5,
    MINIMIZE = False,
    SEED = 1.5,
    PARENT_SELECTION_METHOD = 'tournament',
    RANDOM_SEED = 4
  )
  chosen = [int(solution[f'x{index}']) for index in range(1, 11)]
  weight = sum(weight * bit for weight, bit in zip(weights, chosen))
  value = sum(value * bit for value, bit in zip(values, chosen))

  assert weight <= capacity
  assert value == 280
  assert float(result) == 280.0

def test_canonical_algorithm_is_reproducible_with_random_seed() -> None:
  first_population = Population(
    [(0, 2), (0, 2)],
    1,
    1,
    0.05,
    'x y',
    sympify('x + y'),
    sample_size = 5
  )
  second_population = Population(
    [(0, 2), (0, 2)],
    1,
    1,
    0.05,
    'x y',
    sympify('x + y'),
    sample_size = 5
  )

  first_result = first_population.canonical_algorithm(
    ITERATIONS = 5,
    MINIMIZE = False,
    SEED = 1.5,
    PARENT_SELECTION_METHOD = 'roulette',
    CROSSOVER_METHOD = 'uniform',
    MUTATION_METHOD = 'flipping',
    RANDOM_SEED = 123
  )
  second_result = second_population.canonical_algorithm(
    ITERATIONS = 5,
    MINIMIZE = False,
    SEED = 1.5,
    PARENT_SELECTION_METHOD = 'roulette',
    CROSSOVER_METHOD = 'uniform',
    MUTATION_METHOD = 'flipping',
    RANDOM_SEED = 123
  )

  assert first_result == second_result
