from json import dumps
from sympy import *

from evo_tools.population import Population

population = Population(
  [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)],
  1,
  1,
  0.1,
  True
)
initial_data = population.select_initial_data(8)
print(dumps(initial_data, indent = 2), end = '\n\n')

variables = 'x y z w v'
x, y, z, w, v = symbols(variables)
f = x + 2 * y + 3 * z + 4 * w + 5 * v

population.fitness(variables, f)
population.crossover()
new_data = population.get_current_data()
print(dumps(new_data, indent = 2), end = '\n\n')
