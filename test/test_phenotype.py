from sympy import sympify
from unittest.mock import patch

from evo_tools.phenotype import build_individual_if_valid, decode_individual, \
  evaluate_individual_objective
from evo_tools.population import Population


def test_build_individual_if_valid_reuses_decoding_for_valid_binary() -> None:
  population = Population(
    [(0, 2)],
    1,
    1,
    0.01,
    'x',
    sympify('x')
  )
  bits = [population._sub_populations[0].bits]
  individual = build_individual_if_valid(
    '00',
    '00',
    bits,
    population._sub_populations,
    population._precision,
    population._parsed_function,
    population._variables_array
  )

  assert individual is not None
  assert individual.get_numbers() == '[0.0]'
  assert decode_individual(
    individual,
    population._sub_populations,
    population._precision
  ) == (['00'], [0.0])


def test_build_individual_if_valid_returns_none_for_invalid_binary() -> None:
  population = Population(
    [(0, 2)],
    1,
    1,
    0.01,
    'x',
    sympify('x')
  )
  bits = [population._sub_populations[0].bits]

  assert build_individual_if_valid(
    '11',
    '10',
    bits,
    population._sub_populations,
    population._precision,
    population._parsed_function,
    population._variables_array
  ) is None

def test_evaluate_individual_objective_reuses_cached_numbers_without_debug() -> None:
  population = Population(
    [(0, 2)],
    1,
    1,
    0.01,
    'x',
    sympify('x * x')
  )
  individual = build_individual_if_valid(
    '10',
    '11',
    [population._sub_populations[0].bits],
    population._sub_populations,
    population._precision,
    population._parsed_function,
    population._variables_array
  )

  assert individual is not None

  with patch(
    'evo_tools.phenotype.decode_individual',
    side_effect = AssertionError('unexpected decode')
  ):
    assert evaluate_individual_objective(
      individual,
      0,
      population._sub_populations,
      population._precision,
      population._objective_function
    ) == 4.0
