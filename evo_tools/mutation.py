from evo_tools.bin_gray import mutate_n_bits_from_binary_or_gray, mutation_binary_or_gray_by_flipping

def apply_mutation(
  mutation_method: str,
  binary_or_gray: str
) -> str:
  if mutation_method == 'one_point':
    return mutate_n_bits_from_binary_or_gray(binary_or_gray)

  if mutation_method == 'two_points':
    return mutate_n_bits_from_binary_or_gray(binary_or_gray, 2)

  if mutation_method == 'flipping':
    return mutation_binary_or_gray_by_flipping(binary_or_gray)

  raise Exception('Mutation method not allowed')

def validate_mutation_method(mutation_method: str) -> None:
  if mutation_method in ['one_point', 'two_points', 'flipping']:
    return

  raise Exception('Mutation method not allowed')
