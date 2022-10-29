import time
from typing import Tuple
from sympy import sympify

from evo_tools.population import Population

def generate_variables_and_ecuation() -> Tuple[str, str]:
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

    is_correct_input = '\n- Is this ecuation correct: f: '
    ecuation = ''

    for i, variable in enumerate(variables_list):
      current_coefficient = coefficients_list[i]
      current_exponent = exponents_list[i]

      if abs(float(current_coefficient)) != 1:
        if float(current_coefficient) > 0:
          if i != 0:
            ecuation += f' + {current_coefficient} * '
          else:
            ecuation += f'{current_coefficient} * '
        else:
          aux = current_coefficient.split('-')
          ecuation += f' {"- ".join(aux)} * '
      elif float(current_coefficient) == -1:
        ecuation += ' - '
      elif i != 0:
        ecuation += ' + '

      if float(current_exponent) != 1:
        ecuation += f'{variable}'
        ecuation += f'^{current_exponent}' \
          if float(current_exponent) > 0 else f'^({current_exponent})'
      else:
        ecuation += f'{variable}'

    ecuation += f' = {expected_result}'
    is_correct_input += f' {ecuation} [y/n]: '

    is_correct = input(f'{is_correct_input}')

    if ['y', 'Y'].__contains__(is_correct):
      break

  if float(expected_result) > 0:
    ecuation = ecuation.replace('=', '-')
  else:
    ecuation = ecuation.replace('=', '+')

  return variables, ecuation

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
  _print = False
):
  variables, ecuation = generate_variables_and_ecuation()
  precision, ranges = generate_precision_and_ranges(variables)

  population = Population(
    ranges,
    precision,
    crossover_rate,
    mutation_rate,
    variables = variables,
    function = sympify(ecuation),
    _print = False
  )

  print('\n#############################################')
  print('Running the canonical algorithm for: \n')
  print(f'Function: {ecuation} = 0')
  print(f'Precision: {precision}')
  print(f'Ranges: {ranges}')
  print('\n#############################################')

  start = time.time()
  scores, solution, result = population.canonical_algorithm(
    sample_size,
    iterations,
    minimize,
    seed,
    _print
  )
  end = time.time()

  print()
  print('scores       :', scores)
  print('solution     :', solution)
  print('result       :', result)
  print(f'time elapsed : {end - start}s')

  return result

if __name__ == '__main__':
  canonical_algorithm()
