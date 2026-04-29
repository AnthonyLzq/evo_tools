from json import loads
from typing import Dict, List

import pandas as pd

from evo_tools.models import Individual


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
  df = pd.DataFrame(loads(str(current_population)))
  print(df, end = '\n\n')

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
