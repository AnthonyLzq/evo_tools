from json import dumps, loads
from sympy import symbols

from evo_tools.population import Population

variables = 'x y z w v'
x, y, z, w, v = symbols(variables)
f = x + 2 * y + 3 * z + 4 * w + 5 * v

population = Population(
  ranges = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)],
  precision = 1,
  crossover_rate = 1,
  mutation_rate = 0.1,
  variables = variables,
  function = f,
  _print = True
)
initial_population = population.select_initial_population(8)
population.print()
# print(loads(str(initial_population)))

# population.fitness(variables, f)
# population.crossover()
# new_data = population.get_current_population()
# print(dumps(new_data, indent = 2), end = '\n\n')
