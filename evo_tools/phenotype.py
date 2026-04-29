from typing import List, Tuple, Union

from evo_tools.bin_gray import binary_to_float
from evo_tools.helpers import sub_strings_by_array
from evo_tools.models import SubPopulation

def validate_binaries_in_range(
  binaries: List[List[str]],
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> bool:
  for binary in binaries:
    for i, gen in enumerate(binary):
      try:
        binary_to_float(
          gen,
          sub_populations[i].numbers_dict,
          sub_populations[i].rng,
          precision
        )
      except Exception:
        return False

  return True

def decode_binary_segments(
  binaries: List[str],
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> Tuple[List[str], List[float]]:
  decoded_strings: List[str] = []
  decoded_numbers: List[float] = []

  for i, binary in enumerate(binaries):
    decoded_string = binary_to_float(
      binary,
      sub_populations[i].numbers_dict,
      sub_populations[i].rng,
      precision
    )
    decoded_strings.append(decoded_string)
    decoded_numbers.append(float(decoded_string))

  return decoded_strings, decoded_numbers

def chromosome_to_numbers_repr(
  binary_or_gray: str,
  bits: List[int],
  sub_populations: List[SubPopulation],
  precision: Union[float, int]
) -> str:
  binaries = sub_strings_by_array(binary_or_gray, bits)
  decoded_strings, _ = decode_binary_segments(binaries, sub_populations, precision)

  return '[' + ', '.join(decoded_strings) + ']'
