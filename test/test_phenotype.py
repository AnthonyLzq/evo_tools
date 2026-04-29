from sympy import sympify

from evo_tools.phenotype import build_individual_if_valid, decode_individual
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
