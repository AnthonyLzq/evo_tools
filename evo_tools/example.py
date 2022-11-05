import time
from typing import Tuple
from sympy import sympify

from evo_tools.population import Population, ParentSelectionMethods, CrossoverMethods, MutationMethods

def generate_variables_and_equation() -> Tuple[str, str]:
  while True:
    variables = input('Welcome, please ingress your variables space separated (x y z ...): ')
    coefficients = input('- Please, ingress your coefficients space separated: ')
    exponents = input('- Please, ingress the exponent space separated for each variable: ')
    expected_result: float = float(input('- Please, ingress your expected result: '))

    variables_list = variables.split()
    coefficients_list = coefficients.split()
    exponents_list = exponents.split()

    if len(variables_list) != len(coefficients_list):
      raise Exception('Variables size does not match the number of coefficients')

    if len(variables_list) != len(exponents_list):
      raise Exception('Variables size does not match the number of exponents')

    if exponents_list.__contains__(0):
      raise Exception('An exponent can not be zero')

    is_correct_input = '\n- Is this equation correct: f: '
    equation = ''

    for i, variable in enumerate(variables_list):
      current_coefficient = coefficients_list[i]
      current_exponent = exponents_list[i]

      if abs(float(current_coefficient)) != 1:
        if float(current_coefficient) > 0:
          if i != 0:
            equation += f' + {current_coefficient} * '
          else:
            equation += f'{current_coefficient} * '
        else:
          aux = current_coefficient.split('-')
          equation += f' {"- ".join(aux)} * '
      elif float(current_coefficient) == -1:
        equation += ' - '
      elif i != 0:
        equation += ' + '

      if float(current_exponent) != 1:
        equation += f'{variable}'
        equation += f'^{current_exponent}' \
          if float(current_exponent) > 0 else f'^({current_exponent})'
      else:
        equation += f'{variable}'

    equation += f' = {expected_result}'
    is_correct_input += f' {equation} [y/n]: '

    is_correct = input(f'{is_correct_input}')

    if ['y', 'Y'].__contains__(is_correct):
      break

  if float(expected_result) > 0:
    equation = equation.replace('=', '-')
  else:
    equation = equation.replace('=', '+')

  return variables, equation

def generate_precision_and_ranges(variables: str):
  ranges = []

  for variable in variables.split():
    lower_bound = float(
      input(f'\n- Please, ingress the lower bound for the range of {variable}: ')
    )
    upper_bound = float(
      input(f'- Please, ingress the upper bound for the range of {variable}: ')
    )

    if upper_bound < lower_bound:
      raise Exception('Bad range')

    ranges.append((lower_bound, upper_bound))

  precision = float(input('\n- Please, ingress the precision: '))

  if precision < 0 or precision > 1:
    raise Exception('Bad precision')

  return precision, ranges

def canonical_algorithm(
  crossover_rate = 1,
  mutation_rate = 0.1,
  sample_size = 45,
  iterations = 100,
  minimize = True,
  seed = 1.5,
  _print = False,
  parent_selection_method: ParentSelectionMethods = 'fitness_proportionate',
  crossover_method: CrossoverMethods = 'one_point',
  mutation_method: MutationMethods = 'one_point'
):
  variables, equation = generate_variables_and_equation()
  precision, ranges = generate_precision_and_ranges(variables)

  population = Population(
    ranges,
    precision,
    crossover_rate,
    mutation_rate,
    variables = variables,
    function = sympify(equation),
    _print = False
  )

  print('\n#############################################')
  print('Running the canonical algorithm for: \n')
  print(f'Function: {equation} = 0')
  print(f'Precision: {precision}')
  print(f'Ranges: {ranges}')
  print('\n#############################################')

  start = time.time()
  scores, solution, result = population.canonical_algorithm(
    sample_size,
    iterations,
    minimize,
    seed,
    _print,
    parent_selection_method,
    crossover_method,
    mutation_method
  )
  end = time.time()

  print()
  print('scores       :', scores)
  print('solution     :', solution)
  print('result       :', result)
  print(f'time elapsed : {end - start}s')

  return result

# if __name__ == '__main__':
#   canonical_algorithm()
