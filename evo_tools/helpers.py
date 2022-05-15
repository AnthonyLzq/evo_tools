from typing import List

def sub_strings_by_array(s: str, a: List[int]) -> List[str]:
  sub_strings: List[str] = []
  i = 0

  while (len(s) > 0 and i < len(a)):
    current_sub_string = s[:a[i]]
    sub_strings.append(current_sub_string)
    s = s[a[i]:]
    i += 1

  return sub_strings
