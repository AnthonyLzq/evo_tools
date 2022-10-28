import os

def algorithm():
  os.system('pytest test/test_canonical_algorithm.py -v -s')

def bin_gray():
  os.system('pytest test/test_bin_gray.py -v -s')
