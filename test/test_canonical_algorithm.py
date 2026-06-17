from unittest import skip
from unittest.mock import patch
from sympy import sympify

from evo_tools import example
from evo_tools.canonical import _genotype_unique_ratio, \
  _has_best_objective_improved, _should_stop_early, finalize_canonical_result
from evo_tools.models import Individual

LINEAR_EQUATION = '2 * x + y - z - 3'

@patch(
  'evo_tools.example.generate_variables_and_equation',
  return_value = ('x y z', LINEAR_EQUATION)
)
@patch(
  'evo_tools.example.generate_precision_and_ranges',
  return_value = (0.1, [(0, 10), (0, 10), (0, 10)])
)
def test_canonical_algorithm_linear(a, b) -> None:
  _, __, result, ___, time = example.canonical_algorithm()
  print(
    f'\n\t{{ "result": {result}, "time": {time}, "equation": "{LINEAR_EQUATION}", "operation": "minimize" }}',
    end = ' '
  )
  result = abs(result)
  assert round(result, 3) >= 12.5 and round(result, 3) <= 13.5 # type: ignore

QUADRATIC_EQUATION_1 = '2 * x^2 - 2*y - z - 6'

@patch(
  'evo_tools.example.generate_variables_and_equation',
  return_value = ('x y z', QUADRATIC_EQUATION_1)
)
@patch(
  'evo_tools.example.generate_precision_and_ranges',
  return_value = (0.1, [(0, 10), (0, 10), (0, 10)])
)
def test_canonical_algorithm_quadratic(a, b) -> None:
  _, __, result, ___, time = example.canonical_algorithm()
  print(
    f'\n\t{{ "result": {result}, "time": {time}, "equation": "{QUADRATIC_EQUATION_1}", "operation": "minimize" }}',
    end = ' '
  )
  result = abs(result)
  assert round(result, 2) >= 30 and round(result, 2) <= 36.5  # type: ignore

POLYGONAL_EQUATION = '(1000/6931 - w*x/(y*z))^2'

@patch(
  'evo_tools.example.generate_variables_and_equation',
  return_value = ('w x y z', POLYGONAL_EQUATION)
)
@patch(
  'evo_tools.example.generate_precision_and_ranges',
  return_value = (1, [(12, 60), (12, 60), (12, 60), (12, 60)])
)
def test_canonical_algorithm_polygonal(a, b) -> None:
  _, __, result, ___, time = example.canonical_algorithm(sample_size = 40)
  print(
    f'\n\t{{ "result": {result}, "time": {time}, "equation": "{POLYGONAL_EQUATION}", "operation": "minimize" }}',
    end = ' '
  )
  result = abs(result)
  assert round(result, 2) <= 0.1  # type: ignore

SINE_EXPONENTIAL_EQUATION = 'sin(y) * exp((1 - cos(x)) ** 2) + cos(x) * exp((1 - sin(y)) ** 2) + (x + y) ** 2'

@patch(
  'evo_tools.example.generate_variables_and_equation',
  return_value = ('x y', SINE_EXPONENTIAL_EQUATION)
)
@patch(
  'evo_tools.example.generate_precision_and_ranges',
  return_value = (0.01, [(-14, 0), (-7, 0)])
)
def test_canonical_algorithm_sine_and_exponential_1(a, b) -> None:
  _, __, result, ___, time = example.canonical_algorithm(
    mutation_rate = 0.01,
    sample_size = 80,
    parent_selection_method = 'roulette',
    crossover_method = 'uniform',
    mutation_method = 'flipping'
  )
  print(
    f'\n\t{{ "result": {result}, "time": {time}, "equation": "{SINE_EXPONENTIAL_EQUATION}", "operation": "minimize" }}',
    end = ' '
  )
  assert round(result, 2) <= 0 # type: ignore

@patch(
  'evo_tools.example.generate_variables_and_equation',
  return_value = ('x y', SINE_EXPONENTIAL_EQUATION)
)
@patch(
  'evo_tools.example.generate_precision_and_ranges',
  return_value = (0.01, [(-14, 0), (-7, 0)])
)
def test_canonical_algorithm_sine_and_exponential_2(a, b) -> None:
  _, __, result, ___, time = example.canonical_algorithm(
    mutation_rate = 0.01,
    sample_size = 80,
    parent_selection_method = 'tournament',
    crossover_method = 'two_points',
    mutation_method = 'two_points'
  )
  print(
    f'\n\t{{ "result": {result}, "time": {time}, "equation": "{SINE_EXPONENTIAL_EQUATION}", "operation": "minimize" }}',
    end = ' '
  )
  assert round(result, 2) <= 0 # type: ignore

QUADRATIC_EQUATION_2 = 'x*x'

@patch(
  'evo_tools.example.generate_variables_and_equation',
  return_value = ('x', QUADRATIC_EQUATION_2)
)
@patch(
  'evo_tools.example.generate_precision_and_ranges',
  return_value = (pow(10, -10), [(0, 31)])
)
def test_canonical_algorithm_quadratic_2(a, b) -> None:
  _, __, result, ___, time = example.canonical_algorithm(
    mutation_rate = 0.01,
    sample_size = 80,
    parent_selection_method = 'tournament',
    minimize = False
  )
  print(
    f'\n\t{{ "result": {result}, "time": {time}, "equation": "{QUADRATIC_EQUATION_2}", "operation": "maximize" }}',
    end = ' '
  )
  result = abs(result)
  assert round(result, 2) <= 31 * 31 and round(result, 2) >= 30 * 30  # type: ignore

def test_finalize_canonical_result_uses_cached_numbers() -> None:
  best_individual = Individual(
    'invalid',
    'invalid',
    0,
    [2],
    '[2.0]',
    sympify('x * x'),
    ['x']
  )

  solution, function = finalize_canonical_result(
    best_individual,
    sympify('x * x'),
    ['x'],
    1,
    0.0,
    [4.0]
  )

  assert solution == {'x': 2.0}
  assert float(function) == 4.0

def test_best_objective_improvement_respects_direction_and_tolerance() -> None:
  assert _has_best_objective_improved(0.99, 1.0, True, 0.001)
  assert not _has_best_objective_improved(0.9995, 1.0, True, 0.001)
  assert _has_best_objective_improved(1.01, 1.0, False, 0.001)
  assert not _has_best_objective_improved(1.0005, 1.0, False, 0.001)

def test_early_stopping_requires_min_iterations_and_patience() -> None:
  assert not _should_stop_early(True, 3, 4, 10, 2)
  assert not _should_stop_early(True, 4, 4, 1, 2)
  assert _should_stop_early(True, 4, 4, 2, 2)
  assert not _should_stop_early(False, 4, 4, 2, 2)

def test_genotype_unique_ratio_counts_distinct_binaries() -> None:
  population = [
    Individual('00', '00', 0, [2], '[0.0]', sympify('x'), ['x']),
    Individual('01', '01', 0, [2], '[1.0]', sympify('x'), ['x']),
    Individual('01', '01', 0, [2], '[1.0]', sympify('x'), ['x'])
  ]

  assert _genotype_unique_ratio(population) == 2 / 3

def test_diversity_aware_early_stopping_requires_low_diversity() -> None:
  assert not _should_stop_early(True, 4, 4, 2, 2, True, 0.5, 0.25)
  assert _should_stop_early(True, 4, 4, 2, 2, True, 0.25, 0.25)
