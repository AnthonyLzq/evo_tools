import random
from typing import Union

import numpy as np


def seed_random_generators(random_seed: Union[int, None]) -> None:
  if random_seed is None:
    return

  random.seed(random_seed)
  np.random.seed(random_seed)
