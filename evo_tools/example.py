from population import Population

population = Population([(-3, 3), (0, 3)], 0, True)
population.print()
population.select_initial_data(4)
population.crossover((1, 2))
population.mutation()