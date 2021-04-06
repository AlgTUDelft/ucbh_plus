# UCB-H+
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the source code and data for the experiments presented in 
[_Generalized Optimistic Q-Learning with Provable Efficiency_](http://www.ifaamas.org/Proceedings/aamas2020/pdfs/p913.pdf).

## How to use

Run `python3 main.py Replacement-v0 -v` and `python3 main.py Lake-v0 -v` to get the results from the paper.
To see the full list of parameters, run `python3 main.py -h`. If any parameters are missing, the defaults
from the [defaults file](defaults.yml) are used instead.

By default, no information is printed to console, and the results are saved to `./results`.
You can change this using a verbosity flag `-v`. Use `-v`, `-vv`, etc. to control how much information you
want to be printed.

If you want to apply the methods to your custom environment, you can see how the agents are used in `run.py`.

## Contents

- `./agent` contains classes for the three agents as well as abstract base classes:
    - `q_ucb_h_learning.py` for UCB-H,
    - `q_ucb_h_plus_learning.py` for UCB-H+,
    - `simple_q_learning_agent.py` for Q-Learning.
- `./environment` contains two environments used in the paper; The versions of the environments from the paper
are registered in OpenAI Gym to use via `gym.make()`, see `./environment/__init__.py`.
    - `frozen_lake` is the adjustable FrozenLake environment that allows to change the slipping probability.
    The registered version is `Lake-v0`.
    - `replacement` is the Replacement environment. The registered version is `Replacement-v0`.
- `./process_results` is a collection of helper methods for saving and plotting the data.
- `./results` is the default directory to save the experiments data to.

## Citation

Please cite the paper if you use it:

```
@inproceedings{Neustroev2020,
  title     = {Generalized Optimistic {Q-Learning} with Provable Efficiency},
  author    = {Neustroev, Grigory and de~Weerdt, Mathijs M.},
  booktitle = {International Conference on Autonomous Agents and Multi-Agent Systems},
  year      = {2020},
  address   = {Auckland, New Zealand},
  publisher = {IFAAMAS},
  month     = {May},
  pages     = {913--921}
}
```
