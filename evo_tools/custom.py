from typing import Generator, Union

def custom_range(
  start: Union[float, int],
  stop: Union[float, int, None] = None,
  step: Union[float, int] = 1.0
) -> Generator[Union[float, int], None, None]:
  """
  Creates a generator in the same way that range does, but this function allows
  decimal steps.

  Args:
    start (Union[float, int]):
      Lower bound of the range.
    stop (Union[float, int, None], optional):
      Upper bound of the range, if not present, the start will be the lower bound
      and the start will be 0.0. Defaults to None.
    step (Union[float, int, None], optional):
      Step to iterate over the range. Defaults to 1.0.

  Yields:
    Generator[Union[float, int], None, None]
  """
  if stop == None:
    stop = start + 0.0
    start = 0.0

  current_value = start

  while True:
    if step > 0 and current_value > stop:
      break
    elif step < 0 and current_value < stop:
      break

    yield current_value

    current_value += step
