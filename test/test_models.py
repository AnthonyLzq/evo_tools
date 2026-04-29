from unittest.mock import patch

from sympy import sympify

from evo_tools.models import Individual


def test_individual_get_fitness_reuses_parsed_numbers() -> None:
  individual = Individual(
    '00',
    '00',
    0,
    [2],
    '[2.0]',
    sympify('x * x'),
    ['x']
  )

  with patch('evo_tools.models.loads', side_effect = AssertionError('unexpected loads')):
    assert individual.get_fitness() == 4.0
