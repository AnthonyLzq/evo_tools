# Evo tools

This package has the goal to implement a general canonical [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm).

## Requirements

- Python v3.10.7 (minimal)
- Pip v22.3 (minimal)

## Installation and usage

### Third part library

If you are using this package as a third part library, you only need to install it. For example with pip would be:

```bash
pip install evo-tools
```

### Play locally

If you want to install it locally and play around with it, please create a virtual environment, to do that you can run:

```bash
python -m venv venv
```

Then activate the virtual environment:

```bash
source venv/bin/activate
```

Finally, install the dependencies with:

```bash
python -m pip install --editable .
```

### Testing

If you want to run the tests available, you can run:

```bash
pytest test -v
```
