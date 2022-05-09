from typing import Generator, Union

def custom_range(
  start: Union[float, int],
  stop: Union[float, int, None] = None,
  step: Union[float, int, None] = None
) -> Generator[Union[float, int], None, None]:
  if stop == None:
    stop = start + 0.0
    start = 0.0

  if step == None:
    step = 1.0

  current_value = start

  while True:
    if step > 0 and current_value > stop:
      break
    elif step < 0 and current_value < stop:
      break

    yield current_value

    current_value += step
