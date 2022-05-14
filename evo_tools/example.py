from json import dumps

from bin_gray import float_to_binary_and_gray, range_of_numbers_binary_and_gray
from population import Population

population = Population([(-3, 3), (1, 3)], 0.1, True)
population.print()
initial_data = population.select_initial_data(10)
print(initial_data)
# population.crossover((1, 2))
# population.mutation()

# n = 0.001
# rng = [-1, 0.1]
# precision = 0.001

# b_number, _ = float_to_binary_and_gray(n, rng, precision)
# print(b_number)

# rng = [-2, 3]
# precision = 0.1

# numbers = range_of_numbers_binary_and_gray(rng, precision)
# print(dumps(numbers, indent = 2))
