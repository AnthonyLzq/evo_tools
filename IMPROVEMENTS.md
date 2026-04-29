# IMPROVEMENTS

## Purpose

This document turns the technical assessment of the project into an actionable implementation plan. The goal is to use it as a working guide to modernize `evo_tools` without mixing incompatible changes or breaking existing behavior more than necessary.

## Executive summary

The project already has real value, but it is currently held back by five groups of issues:

1. **Algorithmic correctness**: there are bugs in the genotype-phenotype layer, mutation, scoring, and selection.
2. **Excessive coupling**: `population.py` concentrates too much behavior and mixes domain logic, operators, engine flow, and debug output.
3. **High execution cost**: the code does too much symbolic work and too much parsing / string reconstruction.
4. **Outdated packaging**: the project still uses a legacy `setup.py` layout with incomplete metadata and dependency management.
5. **Fragile QA**: the tests are not reproducible, some dependencies are missing, and parts of the suite no longer represent the current implementation.

## Improvement goals

By the end of this roadmap, the project should meet the following goals:

- The algorithm should behave correctly for both **minimization** and **maximization**.
- Conversion between **real ranges**, **binary**, and **Gray code** should be deterministic and verifiable.
- The genetic engine should be split into smaller, typed, testable modules.
- Fitness evaluation should move from a mostly symbolic approach to a predominantly numeric one.
- The package should install and test using a modern Python workflow (`pyproject.toml`, extras, pytest, CI).
- The repository should no longer version generated files or depend on unsafe test runner scripts.

## Implementation principles

1. **Correctness first, performance second**.
2. **One phase at a time**: do not mix large refactors with delicate algorithmic fixes.
3. **Each phase must leave the repository in a runnable state**.
4. **Before splitting files, lock expected behavior with deterministic tests**.
5. **Do not keep adding features** until the core engine is stable.

## Recommended execution order

| Phase | Goal | Expected outcome |
|---|---|---|
| 0 | Hygiene and baseline | Clean repository, reproducible environment, no junk artifacts |
| 1 | Encoding / decoding correctness | Deterministic and consistent genotype-phenotype mapping |
| 2 | Fitness and selection correctness | Correct minimization / maximization behavior |
| 3 | Engine refactor | More modular and maintainable architecture |
| 4 | Optimization | Lower per-iteration cost and less symbolic overhead |
| 5 | Packaging, tests, and CI | Modern, distributable, contributor-friendly project |

---

## Phase 0 - Hygiene and baseline

### Goal

Clean the repository and establish a reliable baseline before touching more delicate logic.

### Issues identified

- The repository versions `__pycache__` folders and `.pyc` files.
- There are tracked artifacts from modules that no longer exist.
- `.gitignore` is incomplete.
- The test environment is not declared in a modern, reproducible way.

### Files involved

- `.gitignore`
- `setup.py`
- `README.md`
- tracked `__pycache__` artifacts

### Step by step

1. **Remove versioned generated artifacts**
   - Remove every tracked `__pycache__/` directory and `*.pyc` file from the repository.
   - Confirm whether old modules (`custom`, `example2`, etc.) left stale compiled artifacts and delete them from version control.

2. **Harden `.gitignore`**
   - Add:
     - `__pycache__/`
     - `*.pyc`
     - `.pytest_cache/`
     - `.coverage`
     - `htmlcov/`
     - `.venv/`
     - any local tooling folders that should not be versioned

3. **Define the environment baseline**
   - Decide the minimum supported Python version.
   - Align README, metadata, and local development instructions with that version.

4. **Define the development install flow**
   - Establish a single flow for contributors:
     - create a virtual environment
     - install dependencies
     - run tests

### Acceptance criteria

- The repository no longer versions generated files.
- A clean environment can install dependencies without ambiguous steps.
- The basic documentation clearly explains how to set up the project.

### Risks / notes

- This phase should not change functional behavior; it only prepares the ground.

---

## Phase 1 - Encoding / decoding correctness and domain representation

### Goal

Fix the most critical layer of the algorithm: how ranges are discretized and how values are translated between real numbers, binary, and Gray code.

### Issues identified

1. `binary_to_float()` contains an incorrect formula for several decimal precision cases.
2. `range_of_numbers_binary_and_gray()` no longer represents the full range; it currently returns a random sample.
3. The name, docstring, and tests do not match the current behavior.
4. Validation depends on a partial `numbers_dict` plus a flawed fallback path.

### Files involved

- `evo_tools/bin_gray.py`
- `test/test_bin_gray.py`
- `README.md`

### Design decision to close first

The project needs to choose one of these two models and enforce it consistently:

1. **Model A - fully materialized discrete domain**
   - Generate every valid discrete value in the range.
   - Easier to reason about and test.
   - Uses more memory for large domains.

2. **Model B - implicit formula-based domain**
   - Do not store a full table.
   - Real <-> binary conversion is done mathematically.
   - Scales better, but requires a cleaner implementation.

### Recommendation

Use **Model B** for the main engine, and keep explicit full-domain generation only for tests or inspection utilities.

### Step by step

1. **Redefine discretization semantics**
   - Document exactly what `precision` means.
   - Document how many valid discrete values exist for a given range.
   - Define whether the range bounds are inclusive.

2. **Fix `number_of_bits_for_a_range()`**
   - Revisit the current formula.
   - Replace it with a mathematically clear implementation.
   - Remove magic constants and unnecessary double-rounding.
   - Add edge-case tests.

3. **Fix `binary_to_float()`**
   - Remove the flawed fallback formula.
   - Make translation strictly consistent with the chosen discretization model.
   - If a binary string is invalid, raise an explicit, descriptive exception.

4. **Redesign `range_of_numbers_binary_and_gray()`**
   - If the function stays, it must do exactly one of these:
     - generate the full domain,
     - or generate a sample with a new name and new documentation.
   - Do not keep a function named as a full "range" generator if it returns a random sample.

5. **Make representation deterministic**
   - If sampling is still needed anywhere, inject an RNG or explicit seed.
   - Remove hidden dependencies on global `random` state.

6. **Review Gray code formatting**
   - Verify that bit length is preserved.
   - Align `binary_to_gray()` and `gray_to_binary()` so they do not rely on downstream fixes.

7. **Add strong tests**
   - Roundtrip tests:
     - integer -> binary -> Gray -> binary -> integer
   - Roundtrip with range and precision:
     - discrete real value -> binary -> discrete real value
   - Edge cases:
     - lower bound
     - upper bound
     - precision = 1
     - negative ranges

### Acceptance criteria

- Conversion is deterministic and does not depend on partial lookup tables.
- Roundtrip tests pass.
- Documentation and implementation say the same thing.

### Risks / notes

- This phase may break assumptions inside `Population`; avoid large architectural refactors until this is stable.

---

## Phase 2 - Fitness, scoring, minimization, and selection correctness

### Goal

Make the engine optimize correctly in both modes: minimize and maximize.

### Issues identified

1. The current scoring favors low objective values even when the problem is a maximization task.
2. `roulette` and `fitness_proportionate` inherit that inconsistency.
3. `tournament` uses `abs(fitness)`, which distorts the actual objective.
4. The score-based stop condition can almost never trigger.

### Files involved

- `evo_tools/population.py`
- `test/test_canonical_algorithm.py`

### Design decision to close first

These concepts need to be separated clearly:

- **objective value**: the raw value of the objective function
- **fitness score**: the transformed value used for selection
- **best ordering**: the rule used to rank individuals

Right now those concepts are mixed together.

### Recommendation

Use an explicit model:

- store the real **objective value**;
- derive a positive **fitness score** only for the methods that need it;
- rank and choose the "best individual" using a clear rule based on `minimize`.

### Step by step

1. **Separate objective and score**
   - Each individual should expose:
     - objective value
     - fitness score
   - Do not use a single `_score` field to represent both.

2. **Rewrite `_fitness()`**
   - First compute the raw objective value.
   - Then derive a score consistent with the selection method.
   - Do not mix raw function value with transformed fitness.

3. **Define the probabilistic selection fitness strategy**
   - It must work with both negative and positive objective values.
   - It must be consistent for minimization and maximization.
   - Document the transformation.

4. **Fix `roulette` and `fitness_proportionate`**
   - Verify that they use probabilities derived from valid fitness values.
   - Avoid negative or degenerate probability distributions.

5. **Fix `tournament`**
   - Remove `abs(...)`.
   - Compare by real objective value while respecting `minimize`.

6. **Fix stop conditions**
   - Define whether the engine stops because of:
     - target objective value reached,
     - stagnation,
     - selection strength,
     - fixed iteration count.
   - Remove conditions that can never trigger.

7. **Make the algorithm tests deterministic**
   - Fix seeds.
   - Use more robust assertions that depend less on raw randomness.

### Acceptance criteria

- Minimization and maximization behave consistently.
- Selection methods do not favor the wrong individuals.
- Stop conditions are understandable and actually reachable.

### Risks / notes

- This phase is delicate because it changes engine semantics, not just implementation details.

---

## Phase 3 - Engine refactor and coupling reduction

### Goal

Split the genetic engine into smaller, more maintainable, more testable pieces.

### Issues identified

- `population.py` has 1272 lines.
- It mixes domain models, operators, selection strategy, evaluation, debug output, and informal serialization.
- It uses strings as internal data structures.

### Files involved

- `evo_tools/population.py`
- new modules to create
- `evo_tools/__init__.py`
- tests

### Proposed module split

- `evo_tools/models.py`
  - `Individual`
  - `SubPopulation`
- `evo_tools/encoding.py`
  - binary / Gray helpers and discretization logic
- `evo_tools/selection.py`
  - roulette, fitness proportionate, tournament
- `evo_tools/crossover.py`
  - one point, two points, uniform
- `evo_tools/mutation.py`
  - one point, two points, flipping
- `evo_tools/engine.py`
  - canonical algorithm orchestration
- `evo_tools/cli.py` or similar
  - interactive layer / commands

### Step by step

1. **Define typed models**
   - Replace strings like `"float;binary;gray"` with explicit structures.
   - Evaluate using `dataclasses`.

2. **Extract pure helpers first**
   - Move stateless functions into dedicated modules.
   - Keep temporary compatibility imports from the old location if needed.

3. **Separate operators**
   - Each crossover and mutation operator should be a small function or class that can be tested in isolation.

4. **Separate the engine**
   - The canonical algorithm should orchestrate:
     - initialization
     - evaluation
     - selection
     - crossover
     - mutation
     - stopping
   - It should not contain the detailed implementation of each operator.

5. **Reduce informal serialization**
   - Remove patterns such as:
     - manually assembled list strings
     - `json.loads(str(...))`
   - Replace them with actual data structures.

6. **Review the public API**
   - `__init__.py` currently re-exports a lot.
   - Define a minimal, stable public API.

### Acceptance criteria

- `population.py` is no longer the monolithic center of the project.
- Operators can be tested independently.
- The public API is clearer and smaller.

### Risks / notes

- This phase should happen only after Phases 1 and 2 have stabilized expected behavior.

---

## Phase 4 - Performance optimization

### Goal

Reduce the per-iteration cost and the unnecessary overhead of the engine.

### Issues identified

- Too much expression reconstruction with `sympy`.
- Repeated use of `sympify(str(...))`.
- Symbolic substitutions for every individual.
- Repeated string concatenation and parsing.
- Repeated validation in crossover and mutation loops.

### Files involved

- `evo_tools/population.py` or the refactored engine modules
- `evo_tools/example.py`
- performance tests if they are added

### Step by step

1. **Compile the objective function**
   - Convert the expression into a numeric callable once.
   - The natural option is `sympy.lambdify()`.

2. **Avoid repeated reconstruction**
   - Do not call `sympify(str(self._function))` per individual or per iteration.
   - Prepare the function once when building the engine.

3. **Optimize population evaluation**
   - If it makes sense, evaluate individuals with lighter numeric structures.
   - Separate objective value computation from summary statistics computation.

4. **Reduce string churn**
   - Move to lists, tuples, dataclasses, and controlled joins.

5. **Optimize child validation**
   - Revisit the crossover retry flow.
   - Avoid loops where every attempt re-decodes everything from scratch unless it is truly needed.

6. **Centralize RNG**
   - Inject an explicit random generator.
   - Do not mix `random` and `numpy.random` without control.

7. **Review heavy dependencies**
   - Confirm whether `pandas` is really needed as a runtime dependency.
   - If it is only used for debug output, move it out of the runtime path or remove it.

### Acceptance criteria

- Per-iteration evaluation is simpler.
- The amount of repeated symbolic work is reduced.
- Reproducibility improves after centralizing RNG.

### Risks / notes

- Do not optimize before correctness is closed; otherwise the project only speeds up incorrect behavior.

---

## Phase 5 - Packaging, tests, and CI modernization

### Goal

Turn the project into a modern Python package that is easy to install, test, and maintain.

### Issues identified

- Legacy `setup.py` layout without `pyproject.toml`.
- Obsolete `pytest-runner`.
- Incomplete `tests_require`.
- Installable scripts that run tests via `os.system`.
- Missing CI and modern pytest configuration.

### Files involved

- `pyproject.toml`
- `setup.py`
- `README.md`
- `test/setup.py`
- `test/test_bin_gray.py`
- `test/test_canonical_algorithm.py`
- optional `pytest.ini`
- optional GitHub Actions workflow

### Step by step

1. **Migrate packaging to `pyproject.toml`**
   - Declare a modern build system.
   - Move the main metadata into the new file.
   - Keep `setup.py` minimal or remove it if it is no longer needed.

2. **Declare metadata correctly**
   - `requires-python`
   - MIT license
   - correct project URLs
   - realistic classifiers

3. **Split dependencies by scope**
   - runtime
   - test
   - dev

4. **Remove `pytest-runner`**
   - It should no longer be part of the workflow.

5. **Remove test entry points**
   - Replace `test_algorithm` and `test_bin_gray`.
   - If a CLI is desired, create a real package CLI instead.

6. **Modernize the test suite**
   - Add pytest configuration.
   - Fix seeds.
   - Add missing dependencies such as `scipy`, or replace them if they are unnecessary.
   - Review brittle assertions and any assertions that no longer match current behavior.

7. **Add CI**
   - Create a workflow that runs:
     - installation
     - tests
     - basic validations

8. **Update README**
   - modern installation flow
   - basic library usage
   - local development setup
   - how to run tests

### Acceptance criteria

- The project installs using a modern workflow.
- Tests and documentation match the implementation.
- A new contributor can clone, install, and run the suite without ambiguity.

### Risks / notes

- If the API changes during the refactor, README and tests must be updated at the end of each phase, not only at the very end of the roadmap.

---

## Recommended execution guide

## Step 1

Complete **Phase 0** and leave the repository clean.

## Step 2

Tackle **Phase 1** and close the correct encoding / decoding semantics first.

## Step 3

Tackle **Phase 2** and lock the correct minimize / maximize behavior with tests.

## Step 4

Once correctness is stable, move on to **Phase 3** and perform the larger refactor.

## Step 5

With a cleaner architecture, move on to **Phase 4** and optimize the engine.

## Step 6

Close with **Phase 5** so packaging, tests, and CI are fully modernized.

---

## Master checklist

- [ ] Remove generated artifacts and harden `.gitignore`
- [ ] Define a reproducible development baseline
- [ ] Fix range discretization and bit sizing
- [ ] Fix `binary_to_float()`
- [ ] Decide and document the discrete domain model
- [ ] Fix multi-bit mutation
- [ ] Fix scoring and selection for minimize / maximize
- [ ] Remove `abs()` from tournament selection
- [ ] Fix stop conditions
- [ ] Make algorithm tests deterministic
- [ ] Split `population.py`
- [ ] Replace serialized strings with typed structures
- [ ] Compile the objective function and reduce symbolic overhead
- [ ] Centralize RNG
- [ ] Migrate to `pyproject.toml`
- [ ] Modernize pytest and test dependencies
- [ ] Remove entry points that run tests
- [ ] Add CI
- [ ] Update README with the new workflow

---

## Strategy notes

- It is best to work **one phase per commit or per logical PR**.
- If Phase 1 or 2 reveals a strong incompatibility with the current API, prioritize correctness and keep a temporary compatibility layer.
- If the long-term goal is to make `evo_tools` a reusable library, the interactive layer in `example.py` should stop being the main entry point.

## Recommended next move

Start with **Phase 0 + Phase 1**. Those two phases have the highest leverage and unblock the rest of the roadmap.
