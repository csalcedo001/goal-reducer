# Goal Reducer with Loop Removal

This repo contains code for *[Goal Reduction with Loop-Removal Accelerates RL and Models Human Brain Activity in Goal-Directed Learning (NeurIPS 2024 spotlight poster)](https://neurips.cc/virtual/2024/poster/94732)*


## Prerequisites

We used Python 3.10 and CUDA 12.0.

To run this project, you'll need to install dependencies with 

```bash
pip install -r requirements.txt
```

You may also need to compile a cython extension to get faster replay buffer sampling efficiency:

```bash
. compile.sh
```

## Quickstart


To run the basic comparison with the four-room task, you can run the following command:

```bash
. run_gridworld.sh
```

Other experiments follow a similar pattern, see the corresponding flags in `run_gridworld.py`, `run_robot_arm.py`, `run_treasure_hunting.py` and `run_fmri.py`.

