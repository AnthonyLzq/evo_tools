import pytest
from evo_tools import bin_gray
from math import log, log2
from scipy.spatial.distance import hamming
from typing import Tuple, Union

@pytest.mark.parametrize(
  'binary, integer',
  [
    ('10', 2),
    ('1011', 11),
    ('0', 0)
  ]
)
def test_binary_to_int(binary: str, integer: int) -> None:
  assert bin_gray.binary_to_int(binary) == integer

@pytest.mark.parametrize(
  'binary, gray',
  [
    ('0', '0'),
    ('1', '1'),
    ('10', '11'),
    ('11', '10'),
    ('110', '101'),
    ('1011', '1110'),
    ('10010111011', '11011100110')
  ]
)
def test_binary_to_gray(binary: str, gray: str) -> None:
  assert bin_gray.binary_to_gray(binary) == gray

@pytest.mark.parametrize(
  'integer, binary',
  [
    (2, '10'),
    (11, '1011'),
    (0, '0'),
    (1211, '10010111011')
  ]
)
def test_int_to_binary(integer: int, binary: str) -> None:
  assert bin_gray.int_to_binary(integer) == binary

@pytest.mark.parametrize(
  'integer, gray',
  [
    (2, '11'),
    (11, '1110'),
    (0, '0'),
    (1211, '11011100110')
  ]
)
def test_int_to_gray(integer: int, gray: str) -> None:
  assert bin_gray.int_to_gray(integer) == gray

@pytest.mark.parametrize(
  'binary, bits, formatted_binary',
  [
    ('11', 3, '011'),
    ('1110', 8, '00001110'),
    ('101', 6, '000101')
  ]
)
def test_format_to_n_bits(
  binary: str,
  bits: int,
  formatted_binary: str
) -> None:
  assert bin_gray.format_to_n_bits(binary, bits) == formatted_binary

@pytest.mark.parametrize(
  'binary, result, distance',
  [
    ('1011011010101', bin_gray.mutate_binary_or_gray('1011011010101'), 1),
    ('1011011101011', bin_gray.mutate_binary_or_gray('1011011101011'), 1),
    ('1101010101011', bin_gray.mutate_binary_or_gray('1101010101011'), 1)
  ]
)
def test_mutate_binary_or_gray(binary: str, result: str, distance: int) -> None:
  assert hamming(list(binary), list(result)) * len(result) == distance

@pytest.mark.parametrize(
  'rng, precision, validate',
  [
    ((0, 5), 0.1, True),
    ((0, 3), 0.1, True),
    ((-1, 4), 0.01, True)
  ]
)
def test_number_of_bits_for_a_range(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int],
  validate: bool
) -> None:
  x0, xf = rng
  p10 = pow(precision, -1) if precision != 1 else 1
  n_decimal_digits = int(round(log(p10, 10)))
  bits = int(round((log2((xf - x0) * pow(10, n_decimal_digits)) + 0.9)))

  assert bits == bin_gray.number_of_bits_for_a_range(rng, precision, validate)

@pytest.mark.parametrize(
  'rng, precision, validate',
  [
    ((0, -5), 0.1, True)
  ]
)
def test_number_of_bits_for_a_range_throw_bad_range_error(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int],
  validate: bool
) -> None:
  x0, xf = rng
  with pytest.raises(
    Exception,
    match = f'Bad range, {xf} must be greater than {x0}.'
  ):
    bin_gray.number_of_bits_for_a_range(rng, precision, validate)

@pytest.mark.parametrize(
  'rng, precision, validate',
  [
    ((0, 5), -0.1, True)
  ]
)
def test_number_of_bits_for_a_range_throw_precision_error_out_of_bounds(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int],
  validate: bool
) -> None:
  with pytest.raises(
    Exception,
    match = 'Precision can be only a positive decimal fraction between <0, 1].'
  ):
    bin_gray.number_of_bits_for_a_range(rng, precision, validate)

@pytest.mark.parametrize(
  'rng, precision, validate',
  [
    ((0, 5), 0.2, True)
  ]
)
def test_number_of_bits_for_a_range_throw_precision_error_precision_not_decimal(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int],
  validate: bool
) -> None:
  with pytest.raises(
    Exception,
    match = f'Bad precision: {precision} must be a positive decimal fraction or 1.'
  ):
    bin_gray.number_of_bits_for_a_range(rng, precision, validate)

@pytest.mark.parametrize(
  'rng, precision',
  [
    ((0, 5), -0.1)
  ]
)
def test_range_of_numbers_binary_and_gray_throw_precision_error_out_of_bounds(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int]
) -> None:
  with pytest.raises(
    Exception,
    match = 'Precision can be only a positive decimal fraction between <0, 1].'
  ):
    bin_gray.range_of_numbers_binary_and_gray(rng, precision)

@pytest.mark.parametrize(
  'rng, precision',
  [
    ((0, 5), 0.3)
  ]
)
def test_range_of_numbers_binary_and_gray_throw_precision_error_precision_not_decimal(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int]
) -> None:
  with pytest.raises(
    Exception,
    match = f'Bad precision: {precision} should be a positive decimal fraction or 1.'
  ):
    bin_gray.range_of_numbers_binary_and_gray(rng, precision)

@pytest.mark.parametrize(
  'rng, precision',
  [
    ((0, 5), 0.1),
    ((0, 3), 0.1),
    ((-1, 4), 0.01)
  ]
)
def test_range_of_numbers_binary_and_gray(
  rng: Tuple[Union[float, int], Union[float, int]],
  precision: Union[float, int]
) -> None:
  x0, xf = rng
  numbers, bits = bin_gray.range_of_numbers_binary_and_gray(rng, precision)

  calculated_bits = bin_gray.number_of_bits_for_a_range(rng, precision, False)
  len_numbers = int((xf - x0) / precision + 1)

  assert len_numbers == len(numbers) and bits == calculated_bits
