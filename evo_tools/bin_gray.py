from random import randint
from math import log, log2
from typing import List, Tuple, Union

from evo_tools.custom import custom_range

class NumberBinaryAndGray():
  def __init__(self, number: str, binary: str, gray: str) -> None:
    self._number = number
    self._binary = binary
    self._gray = gray

  def get_number(self) -> str:
    return self._number

  def get_binary(self) -> str:
    return self._binary

  def get_gray(self) -> str:
    return self._gray

def binary_to_int(b: str) -> int:
  """
  Function to change binary to integer.

  Args:
    b (str):
      Binary expressed in str.

  Returns:
    int: binary expressed in integer.
  """
  return int(b, 2)

def binary_to_gray(b: str) -> str:
  """
  Function to change a binary to gray.

  Args:
    b (str):
      Binary expressed in str.

  Returns:
    str: binary expressed in gray
  """
  n = binary_to_int(b)
  n ^= (n >> 1)

  return bin(n)[2:]

def int_to_binary(n: int) -> str:
  """
  Function to change a integer to binary.

  Args:
    n (int)

  Returns:
    str: integer expressed in binary
  """
  b = bin(n)[2:]

  return b

def int_to_gray(n: int) -> str:
  """
  Function to change a integer to gray

  Args:
    n (int)

  Returns:
    str: integer expressed in gray
  """
  b = bin(n)[2:]
  g = binary_to_gray(b)

  return g

def format_to_n_bits(b_number: str, bits: int) -> str:
  """
  Function to change the format of a binary to a fixed format. It adds 0s at the
  beginning if it is required. For example, if ('101', 4) is the input, then
  '0101' will be the output.

  Args:
    b_number (str):
      Binary number to be formatted expressed in string
    bits (int):
      Minimal length required for the binary

  Returns:
      str: _description_
  """
  l = len(b_number)
  b_number = str(0) * (bits - l) + b_number

  return b_number

def range_of_numbers_binary_and_gray(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int]
) -> Tuple[List[NumberBinaryAndGray], int]:
  """
  Function to create a list of :class:`NumberBinaryAndGray`. For example, given
  a range [-1, 1] and a precision of 0.1, this functions will create an array
  of NumberBinaryAndGray, which is a object that has a number (float representation),
  binary and gray attributes in strings.

  Args:
    rng (Tuple[Union[float, int], Union[float, int]]):
      Closed interval where the list will be created.

    precision (Union[float, int]):
      Decimal fraction in case of floats, 1 if only integers are need it.

  Raises:
    Exception: when the given precision is neither 1 nor a decimal fraction.

  Returns:
    Tuple[List[NumberBinaryAndGray], int]:
      In the first position contains the given range with the three possible
      representations (float, binary and gray).

      In the second position contains the number of bits that are used to represent
      the given float or integer numbers in binary.
  """
  x0, xf = rng

  if precision < 0 or precision > 1:
    raise Exception(
      'Precision can be only a positive decimal fraction between <0, 1]'
    )

  p10 = pow(precision, -1) if precision != 1 else 1

  if p10 != 1 and p10 % 10 != 0:
    raise Exception(f'Bad precision: {precision} should be a positive decimal fraction or 1.')

  n_decimal_digits = int(round(log(p10, 10)))
  bits = int(round((log2((xf - x0) * pow(10, n_decimal_digits)) + 0.9)))
  numbers: List[NumberBinaryAndGray] = []

  for i in custom_range(x0, xf + pow(10, -n_decimal_digits), precision):
    number = int(p10 * i)

    if x0 < 0:
      number += int(-1 * x0 * p10)
    elif x0 > 0:
      number -= int(x0 * p10)

    index = round(i, n_decimal_digits)

    if index <= xf:
      numbers.append(
        NumberBinaryAndGray(
          format(
            index,
            f'.{n_decimal_digits}f'
          ) if index != 0 else str(index * index) + str(0) * (n_decimal_digits - 1),
          format_to_n_bits(int_to_binary(number), bits),
          format_to_n_bits(int_to_gray(number), bits)
        )
      )

  return numbers, bits

def float_to_binary_and_gray(
  n: float,
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: float
) -> Tuple[NumberBinaryAndGray, int, List[NumberBinaryAndGray]]:
  """
  Function to change a float to its binary and gray representation in a given
  range with a given precision.

  Args:
    n (float):
      Float to be changed to binary and gray.

    rng (Tuple[Union[float, int], Union[float, int]]):
      Interval where n presumably belongs.

    precision (float):
      Decimal fraction in case of floats, 1 if only integers are need it.

  Raises:
    Exception:
      When the given precision is neither 1 nor a decimal fraction or when the
      float does not belong to the given range.

  Returns:
    Tuple[NumberBinaryAndGray, int, List[NumberBinaryAndGray]]:
      In the first position a object that has a number (float representation),
      binary and gray attributes in strings.

      In the second position the number of bits that are used to represent
      the given float or integer numbers in binary.

      In the third position the given range with the three possible
      representations (float, binary and gray).
  """
  x0, xf = rng

  if n < x0 or n > xf:
    raise Exception(f'Bad input: {n} is out of bounds: {rng}.')

  if precision < 0 or precision > 1:
    raise Exception(
      'Precision can be only a positive decimal fraction between <0, 1]'
    )

  numbers, bits = range_of_numbers_binary_and_gray(rng, precision)
  aux: List[NumberBinaryAndGray] = list(
    filter(lambda nbg: nbg.get_number() == str(n), numbers)
  )

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
  """
  Function to change a binary to a float in a given range with a given precision.

  Args:
    b (str):
      Binary number to change.

    rng (Tuple[Union[float, int], Union[float, int]]):
      Interval where b presumably belongs.

    precision (float):
      Decimal fraction in case of floats, 1 if only integers are need it.

  Raises:
    Exception: when the binary does not belong to the given range.

  Returns:
    NumberBinaryAndGray: the given range with the three possible representations
    (float, binary and gray).
  """
  numbers, _ = range_of_numbers_binary_and_gray(rng, precision)
  aux = list(filter(lambda nbg: nbg.get_binary() == b, numbers))

  if len(aux) == 0:
    raise Exception(
      f'Bad input: {b} is not in the discrete range: {rng} with precision: {precision}'
    )

  return aux[0]

def mutate_binary_or_gray(b: str) -> str:
  """
  Function to change one bit from a given binary.

  Args:
    b (str)

  Returns:
    str: binary with a bit changed
  """
  length = len(b) - 1
  pos_bit = randint(0, length)
  new_bit = '0' if b[pos_bit] == '1' else '1'

  return b[:pos_bit] + new_bit + b[pos_bit + 1:]
