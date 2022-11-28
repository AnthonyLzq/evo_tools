from unittest import skip
from unittest.mock import patch
from evo_tools import example

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
