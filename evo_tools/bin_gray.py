from random import randint
from math import log, log2
from typing import List, Tuple, Union
from sys import version_info

if version_info >= (3, 8):
  from typing import TypedDict
else:
  from typing_extensions import TypedDict

from evo_tools.custom import custom_range

class NumberBinaryAndGray(TypedDict):
  number: str
  binary: str
  gray: str

def binary_to_int(n: str) -> int:
  return int(n, 2)

def binary_to_gray(n: str) -> str:
  n = binary_to_int(n)
  n ^= (n >> 1)

  return bin(n)[2:]

def gray_to_binary(n: str) -> str:
  n = binary_to_int(n)
  mask = n

  while mask != 0:
    mask >>= 1
    n ^= mask

  return bin(n)[2:]

def int_to_binary(n: int) -> str:
  b = bin(n)[2:]

  return b

def int_to_gray(n: str) -> str:
  b = bin(n)[2:]
  g = binary_to_gray(b)

  return g

def format_to_n_bits(b_number: str, bits: int) -> str:
  l = len(b_number)
  b_number = str(0) * (bits - l) + b_number

  return b_number

def range_of_numbers_binary_and_gray(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int]
) -> List[NumberBinaryAndGray]:
  x0, xf = rng

  if precision < 0 or precision > 1:
    raise Exception(
      'Precision can be only a positive decimal fraction betwen <0, 1]'
    )

  p10 = pow(precision, -1) if precision != 1 else 1
  n_decimal_digits = int(round(log(p10, 10)))
  bits = int(round((log2((xf - x0) * pow(10, n_decimal_digits)) + 0.9)))

  if p10 != 1 and p10 % 10 != 0:
    raise Exception(f'Bad precision: {precision} should be a positive decimal fraction.')

  numbers = []

  for i in custom_range(x0, xf + pow(10, -n_decimal_digits), precision):
    number = int(p10 * i)

    if x0 < 0:
      number += int(-1 * x0 * p10)
    elif x0 > 0:
      number -= int(x0 * p10)

    index = round(i, n_decimal_digits)
    numbers.append({
      'number': format(
        index,
        f'.{n_decimal_digits}f'
      ) if index != 0 else str(index * index) + str(0) * (n_decimal_digits - 1),
      'binary': format_to_n_bits(int_to_binary(number), bits),
      'gray': format_to_n_bits(int_to_gray(number), bits)
    })

  return numbers, bits

def float_to_binary_and_gray(
  n: float,
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: float
) -> str:
  x0, xf = rng

  if n < x0 or n > xf:
    raise Exception(f'Bad input: {n} is out of bounds: {rng}.')

  if precision < 0 or precision > 1:
    raise Exception(
      'Precision can be only a positive decimal fraction betwen <0, 1]'
    )

  numbers, bits = range_of_numbers_binary_and_gray(rng, precision)
  aux = list(filter(lambda number: number['number'] == str(n), numbers))

  if len(aux) == 0:
    raise Exception(
      f'Bad input: {n} is not in the discrete range: {rng} with precision: {precision}'
    )

  return aux[0], bits, numbers

def binary_to_float(
  b: str,
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: float
) -> NumberBinaryAndGray:
  numbers, _ = range_of_numbers_binary_and_gray(rng, precision)
  aux = list(
    filter(
      lambda e: e['binary'] == b,
      numbers
    )
  )

  if len(aux) == 0:
    raise Exception(
      f'Bad input: {b} is not in the discrete range: {rng} with precision: {precision}'
    )

  return aux[0]

def mutate_binary_or_gray(n: str) -> str:
  length = len(n) - 1
  pos_bit = randint(0, length)
  new_bit = '0' if n[pos_bit] == '1' else '1'

  return n[:pos_bit] + new_bit + n[pos_bit + 1:]
