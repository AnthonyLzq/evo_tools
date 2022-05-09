from typing import Generator

def custom_range(
  start: float | int,
  stop: float | int | None = None,
  step: float | int | None = None
) -> Generator[float | int, None, None]:
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
