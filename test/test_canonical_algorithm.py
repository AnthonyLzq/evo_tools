from unittest.mock import patch
from evo_tools import example

@patch(
  'evo_tools.example.generate_variables_and_ecuation',
  return_value = ('x y z', '2 * x + y - z - 3')
)
@patch(
  'evo_tools.example.generate_precision_and_ranges',
  return_value = (0.1, [(0, 10), (0, 10), (0, 10)])
)
def test_canonical_algorithm_linear(a, b) -> None:
  result = abs(example.canonical_algorithm())
  assert round(result, 3) <= 0.015  # type: ignore

@patch(
  'evo_tools.example.generate_variables_and_ecuation',
  return_value = ('x y z', '2 * x^2 - 2*y - z - 6')
)
@patch(
  'evo_tools.example.generate_precision_and_ranges',
  return_value = (0.1, [(0, 10), (0, 10), (0, 10)])
)
def test_canonical_algorithm_cuadratic(a, b) -> None:
  result = abs(example.canonical_algorithm())
  assert round(result, 2) <= 0.021  # type: ignore
