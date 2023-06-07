# Neosocratic

Neosocratic is a CLI (command line interface) for AI prompt engineers to design dialogues and apply the scientific method to responses.

## Installation

You need the GitHub CLI installed and authorized to clone the repository.
- Use `gh` from https://github.com/cli/cli#installation

We use Python 3.10, and I can't guarantee support for other versions.
- Use Miniconda3 (`conda` from https://docs.conda.io/en/latest/miniconda.html)
- `conda env create -n py310 python=3.10` to create a new environment.
- `conda activate py310` to activate the environment.

### Linux and MacOS
```sh
gh repo clone bionicles/neosocratic && cd neosocratic/neosocratic_cli && make cli
```

# Windows Powershell
```ps1
gh repo clone bionicles/neosocratic && cd neosocratic/neosocratic_cli && pip install -e .
```