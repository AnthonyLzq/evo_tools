from sympy import sympify

from evo_tools.population import Population
from evo_tools.reporting import print_initial_population, print_iteration_summary


def test_reporting_summaries_render_population_without_pandas(capsys) -> None:
  population = Population(
    [(0, 2)],
    1,
    1,
    0.01,
    'x',
    sympify('x')
  )
  population.canonical_algorithm(
    ITERATIONS = 1,
    MINIMIZE = False,
    PARENT_SELECTION_METHOD = 'tournament'
  )

  print_initial_population(population._current_population)
  print_iteration_summary(
    1,
    population._best_individual,
    0.0,
    0.01,
    population._current_population
  )
  output = capsys.readouterr().out

  assert 'Initial population:' in output
  assert '1º iteration.' in output
  assert 'binary=' in output
  assert 'fitness=' in output
