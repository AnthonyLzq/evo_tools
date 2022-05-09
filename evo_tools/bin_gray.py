from random import randint
from typing import List

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

def format_to_n_bits(b: str, bits: int) -> str:
  l = len(b)
  b = str(0) * (bits - l) + b

  return b

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
