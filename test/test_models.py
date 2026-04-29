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


def test_individual_caches_bits_metadata_and_owns_input_lists() -> None:
  bits = [1, 2, 3]
  variables = ['x', 'y', 'z']
  individual = Individual(
    '010101',
    '011001',
    0,
    bits,
    '[1.0, 2.0, 3.0]',
    sympify('x + y + z'),
    variables
  )

  bits.append(99)
  variables[0] = 'changed'

  assert individual.get_bits() == [1, 2, 3]
  assert individual.get_total_bits() == 6
  assert '"bits": "[1, 2, 3]"' in str(individual)
  assert individual.get_fitness() == 6.0
