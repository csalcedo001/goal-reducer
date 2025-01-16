# Installation instructions

## Python environment

### Conda

If you have [Miniconda](https://docs.anaconda.com/miniconda/install/) installed, setting up the Python virtual environment is straightforward. Just run the following command

```bash
conda create --name goal-reducer python=3.10
conda activate goal-reducer
```

### venv

First, ensure you have Python 3.10 installed (check your version with `python --version`). Then, create the virtual environment with

```bash
python3.10 -m venv .venv
```

Activate your Python environment according to your operating system.

* Linux/Mac

```bash
source .venv/bin/activate
```

* Windows (PowerShell)

```bash
.\.venv\Scripts\Activate
```


## Package dependencies

All package installation dependencies should be considered in `requirements.txt` so that you install easily just by running the following command

```bash
pip install -r requirements.txt
```

__Note__: If you happen to find a missing or broken dependency, please update  `requirements.txt` to make it work.


## Weights and Biases

By default, the original codebase uses [Weights and Biases](https://wandb.ai/site/) to store performance metrics throughout training. Create an account and setup your local user to be able to log metrics from the project to your online account.


## Verify installation

Check that your installation is successful by running the main script

```bash
. verify_installation.sh
```