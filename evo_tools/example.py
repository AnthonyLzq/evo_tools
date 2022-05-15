from json import dumps
from sympy import *

from bin_gray import binary_to_float, float_to_binary_and_gray, range_of_numbers_binary_and_gray
from helpers import sub_strings_by_array
from population import Population

population = Population([(1, 2), (4, 5)], 0.01, True)
initial_data = population.select_initial_data(10)
print(initial_data, end='\n\n')
# population.mutation()

x, y = symbols('x y')
f = x + y

population.fitness('x y', f)

# n = 0.001
# rng = [-1, 0.1]
# precision = 0.001

# b_number, _, __ = float_to_binary_and_gray(n, rng, precision)
# print(b_number)
# print(binary_to_float(b_number['binary'], rng, precision))

# rng = [-2, 3]
# precision = 0.1

# numbers = range_of_numbers_binary_and_gray(rng, precision)
# print(dumps(numbers, indent = 2))


# print(sub_strings_by_array('01111011100', [6, 5]))
# 011110
# 11100

# print(sub_strings_by_array('0111101110', [5, 5]))
# 01111
# 01110

# print(sub_strings_by_array('001011000001101010', [10, 8]))
# 01111
# 01110
