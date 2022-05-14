from bin_gray import float_to_gray_and_binary
from json import dumps
# from population import Population

# population = Population([(-3, 3), (0, 3)], 0, True)
# population.print()
# population.select_initial_data(4)
# population.crossover((1, 2))
# population.mutation()

n = 0.001
rng = [-1, 0.1]
precision = 0.001

b_number, numbers = float_to_gray_and_binary(n, rng, precision)
print(b_number)
# print(dumps(numbers, indent=2))
