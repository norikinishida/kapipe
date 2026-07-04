# Repository-Specific Rules

## Scope of Work

- The user edits the code.
- Only provide proposed code changes and explanations.
- Do not create or edit files unless explicitly instructed by the user.
- Files may change dynamically. Always check the latest contents of relevant files before answering, suggesting changes, or writing code.

## Directory Structure

- `kapipe/` is a general-purpose library.
- `experiments/*/` contains experimental code, configurations, data, and outputs.

## Python Environments

- Use pyenv and pyenv-virtualenv for Python environments.
- Use an independent virtual environment for each experiment.
- Run `pyenv local <environment_name>` in each experiment directory to create a `.python-version` file.
- Before running commands, move to the relevant experiment directory and verify that the corresponding Python environment is selected.

## Editable Installation

- Install the repository root in editable mode from each experiment's virtual environment.
- With an editable installation, changes to `kapipe/` are immediately reflected in each experiment without reinstallation.

```bash
cd experiments/<experiment_name>
pyenv local <environment_name>
python -m pip install -e ~/projects/kapipe
```

## Dependency Management

- Manage dependencies directly used by the KAPipe library (`kapipe/`) in the root `pyproject.toml`.
- Keep the dependencies in the root `requirements.txt` and `pyproject.toml` consistent.
- Do not include experiment-specific dependencies in the library dependencies.
- Follow the user's instructions for managing experiment-specific dependencies.
- `kapipe.egg-info/` is generated and must not be edited.

## Running Experiments

- Run each experiment from its corresponding `experiments/*/` directory.
- Do not run experiment scripts from the repository root.
- Save experiment outputs according to the existing conventions of each experiment.

## Coding Style

- Add type hints. Example: `list[str]`
- Add clear, detailed, and sufficient comments in English.
- Prefer more comments.
- Minimize abstractions.
- Prefer straightforward, explicit code.

## Python Implementation Rules

- Define `main()` before all other functions in the file.
- Define functions in top-down order. For example, if function `F` calls function `G`, define `F` before `G`.
- Do not create functions such as `parse_args()`. Write command-line argument parsing directly in the `if __name__ == "__main__"` block.
- As a rule, access dictionary fields directly rather than using `get()`. Example: `data[key]`. This ensures that unexpected dictionary structures fail immediately.
