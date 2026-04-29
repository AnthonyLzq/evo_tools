from typing import Dict, List

from evo_tools.models import Individual


def _format_population(individuals: List[Individual]) -> str:
  return '\n'.join(
    [
      f'  {index}. binary={individual.get_binary()} gray={individual.get_gray()} '
      f'numbers={individual.get_numbers()} bits={individual.get_bits()} '
      f'score={individual.get_score()} fitness={individual.get_fitness()}'
      for index, individual in enumerate(individuals, start = 1)
    ]
  )


def print_initial_population(initial_population: List[Individual]) -> None:
  print('\nInitial population:\n')
  print(_format_population(initial_population))
  print()


def print_iteration_summary(
  current_iteration: int,
  best_individual: Individual,
  selection_strength: float,
  elapsed_time: float,
  current_population: List[Individual]
) -> None:
  print(
    f'\n{current_iteration}º iteration.\nBest individual: {best_individual}.\nSelection strength: {selection_strength}.\nTime elapsed: {elapsed_time}s.'
  )
  print(_format_population(current_population), end = '\n\n')

def print_final_summary(
  current_iteration: int,
  best_individual: Individual,
  selection_strength: float,
  fitness_avg_list: List[float],
  solution: Dict[str, float],
  function
) -> None:
  print(
    f'\n\nFinally:\n{current_iteration}º iteration.\nBest individual: {best_individual}.\nSelection strength: {selection_strength}.'
  )
  print('Solution:')
  print(f'  Variables: {solution}')
  print(f'  Evaluation: {function}')
  print(f'  Fitness average: {fitness_avg_list}')
