from json import dumps, loads
from sympy import symbols

from evo_tools.population import Population

variables = 'x y z w v'
x, y, z, w, v = symbols(variables)
f = x + 2 * y + 3 * z + 4 * w + 5 * v - 50

population = Population(
  ranges = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)],
  precision = 0.01,
  crossover_rate = 1,
  mutation_rate = 0.1,
  variables = variables,
  function = f,
  _print = False
)

population.canonical_algorithm(10)
