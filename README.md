# Evo tools

This package has the goal to implement a general canonical [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm).

## Requirements

- Python 3.10+
- pip

## Installation and usage

### Third-party library

If you want to use this package as a library, install it with pip:

```bash
pip install evo-tools
```

### Local development

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the package in editable mode:

```bash
python -m pip install --editable .
```

### Testing

Install the test dependencies:

```bash
python -m pip install pytest pytest-mock scipy
```

Run the full test suite with:

```bash
python -m pytest test -v
```

Run only the `bin_gray` helper tests with:

```bash
python -m pytest test/test_bin_gray.py -v
```

Run only the canonical algorithm tests with:

```bash
python -m pytest test/test_canonical_algorithm.py -v
```

## Canonical algorithm configuration

The canonical algorithm currently supports these parent selection methods:

- `fitness_proportionate`
- `roulette`
- `tournament`

These values are defined in `evo_tools.selection.PARENT_SELECTION_METHODS` and
re-exported from `evo_tools`, so library consumers can import the supported
options instead of relying on undocumented string literals.

```python
from evo_tools import PARENT_SELECTION_METHODS
```
