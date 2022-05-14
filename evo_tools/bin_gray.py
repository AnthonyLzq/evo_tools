from random import randint
from math import log2
from typing import List, Tuple
from custom import custom_range
from json import dumps

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

def float_to_gray_and_binary(
  n: float,
  rng: Tuple[int, int],
  precision: float
) -> str:
  x0, xf = rng
  p10 = pow(precision, -1)
  n_decimal_digits = int(p10 / 10)
  numbers = {}
  bits = int(log2((xf - x0) * pow(10, n_decimal_digits)) + 0.9)

  if x0 > n or n > xf:
    raise Exception(f'Bad input: {n} is out of bounds: {rng}.')

  if p10 % 10 != 0:
    raise Exception(f'Bad precision: {precision} should be a decimal fraction.')

  for i in custom_range(x0, xf, precision):
    number = int(p10 * i)

    if x0 < 0:
      number += int(-1 * x0 * p10)

    index = round(i, n_decimal_digits)
    numbers[index if index != 0 else -1 * index] = {
      'binary': format_to_n_bits(int_to_binary(number), bits),
      'gray': format_to_n_bits(int_to_gray(number), bits)
    }

  if n not in numbers:
    raise Exception(
      f'Bad input: {n} is not in the discrete range: {rng} with precision: {precision}'
    )

  return numbers[n], numbers

def binary_numbers_with_n_bits(n: int, bits = 8) -> List[str]:
  numbers = []

  for i in range(n):
    b = format_to_n_bits(int_to_binary(i), bits)
    numbers.append(b)

  return numbers

def gray_numbers_with_n_bits(n: int, bits = 8) -> List[str]:
  numbers = []

  for i in range(n):
    b = format_to_n_bits(int_to_gray(i), bits)
    numbers.append(b)

  return numbers

def mutate_binary_or_gray(n: str) -> str:
  length = len(n) - 1
  pos_bit = randint(0, length)
  new_bit = '0' if n[pos_bit] == '1' else '1'

  return n[:pos_bit] + new_bit + n[pos_bit + 1:]
