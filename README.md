# CSP-MRNav Source Release

This repository contains a streamlined source-code release intended for public distribution on GitHub. It retains the core implementations of CSP, LMTE, CBF, the multi-robot environment, and MAPPO, while the training parameter configurations are not publicly released at this stage.

## Currently Released Components

* `crowd_sim/social/`: LMTE, CSP metrics, reward functions, action filtering, and the composite social CBF.
* `crowd_sim/wrappers/`: Integration layer connecting CSP/LMTE with the multi-robot environment.
* `crowd_sim/envs/` and `crowd_sim/multi_robot_core.py`: Multi-robot environments and state contracts.
* `training/algo/` and `training/networks/`: MAPPO optimizer, Actor/Critic networks, storage utilities, and vectorized environments.
* `crowd_nav/policy/`: Basic policy interfaces required for environment execution.
* `train_mappo.py` and `training/test_mappo.py`: Training and evaluation entry points, with configurations supplied at runtime.
* `tests/`: Core unit tests that do not depend on private training parameters.

## Components Not Currently Released

* `crowd_nav/configs/config.py`, `config_mappo.py`, and other training configuration files.
* Checkpoints, model weights, datasets, logs, runtime directories, and configuration snapshots.
* Manuscripts, result figures, experimental exports, internal implementation records.
* Third-party baseline implementations, including AVOCADO, CEMRRL, HeR-DRL, and RL-RVO.
* PyBullet/TurtleBot models and textures; these assets will not be included in the public repository until their redistribution licenses have been verified.

Therefore, the current release allows inspection and testing of the core methodology, but end-to-end training results cannot be reproduced until the required configurations and simulation assets are provided.

## Using Private Configurations Locally

Place the private `config.py` and `config_mappo.py` files under `crowd_nav/configs/`. These files are excluded by `.gitignore`; do not use `git add -f` to force them into the repository.

The configuration module must provide a `Config` class. The training entry point loads the configuration dynamically at runtime, so `python train_mappo.py --help` remains available even when the public repository does not contain the private configuration files.

```powershell
python train_mappo.py --config-module crowd_nav.configs.config_mappo --cpu
```

For evaluation, `training/test_mappo.py` reads the private configuration snapshot associated with the checkpoint only from `<model_dir>/configs/`:

```powershell
python -m training.test_mappo --checkpoint <model_dir>/checkpoints/<checkpoint>.pt --cpu
```

## Environment Setup

The original development environment was validated with Python 3.8. First, install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

In addition, [OpenAI Baselines](https://github.com/openai/baselines) and [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) must be installed separately. Their source code is not redistributed in this repository.

## Testing

```powershell
python -m pytest -q
```

The provided tests cover the publicly released core modules only. They do not constitute reproduction of the benchmarks reported in the paper and do not validate training performance under the unreleased private configurations.

## Pre-Release Checklist

1. Create a new standalone Git repository under `csp/` without inheriting the history of the current working directory.
2. Verify with `git status --short --ignored` that configuration files, model weights, datasets, and logs are properly ignored.
3. Run a secret scanner on all files to be committed, and manually inspect them for absolute paths, IP addresses, usernames, passwords, and other sensitive information.
4. Verify the redistribution licenses of the PyBullet/TurtleBot assets before adding them according to the directory structure referenced by the source code.


