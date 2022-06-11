from sympy import symbols
import time

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

start = time.time()
scores, solution, result = population.canonical_algorithm(100, PRINT = True)
end = time.time()

print()
print('scores       :', scores)
print('solution     :', solution)
print('result       :', result)
print(f'time elapsed: {end - start}s')
