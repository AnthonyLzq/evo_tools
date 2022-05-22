from typing import List
from functools import reduce

def sub_strings_by_array(s: str, l: List[int]) -> List[str]:
  """
  Function to split a string based in a list of sizes.
  For example, consider the following str: '101101101110'
  and the following list: [2, 3, 5, 2]. Then the output will be:
  ['10', '110', '11011', '10'].

  Args:
    s (str):
      String to be splitted

    a (List[int]):
      List of sizes to split the str.

  Raises:
    Exception: if the length of the str is different from the sum of values
    from the list.

  Returns:
    List[str]:
      The splitted str.
  """
  sub_strings: List[str] = []
  i = 0
  length = reduce(lambda a, b: a + b, l)

  if (length != len(s)):
    raise Exception(
      'The sum of the values in the array must be the same length of the array'
    )

  while (len(s) > 0 and i < len(l)):
    current_sub_string = s[:l[i]]
    sub_strings.append(current_sub_string)
    s = s[l[i]:]
    i += 1

  return sub_strings
