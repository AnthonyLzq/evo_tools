from setuptools import setup, find_packages

VERSION = '1.4.0'
DESCRIPTION = 'Evolutionary programming tools'
LONG_DESCRIPTION = 'A package that allows you to implement evolutionary algorithms.'

# Setting up
setup(
  name='evo_tools',
  version=VERSION,
  author='AnthonyLzq (Anthony Luzquiños)',
  author_email='<sluzquinosa@uni.pe>',
  description=DESCRIPTION,
  long_description_content_type='text/markdown',
  long_description=LONG_DESCRIPTION,
  packages=find_packages(),
  install_requires=['sympy', 'numpy'],
  keywords=['python', 'evolutionary programming'],
  project_urls={
    'Source Code': 'https://gitlab.com/AnthonyLzq/evo_tools'
  },
  entry_points={
    'console_scripts': [
      'test_algorithm=test.setup:algorithm',
      'test_bin_gray=test.setup:bin_gray'
    ]
  },
  setup_requires=['pytest-runner'],
  tests_require=['pytest', 'pytest-mock'],
  classifiers=[
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Operating System :: Unix',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows'
  ]
)