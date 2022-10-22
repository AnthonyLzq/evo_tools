import pytest
from evo_tools import bin_gray
from scipy.spatial.distance import hamming

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
