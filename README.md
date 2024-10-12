# Goal Reducer with Loop Removal

This repo contains code for *[Goal Reduction with Loop-Removal Accelerates RL and Models Human Brain Activity in Goal-Directed Learning (NeurIPS 2024 spotlight poster)](https://neurips.cc/virtual/2024/poster/94732)*


## Prerequisites

We used Python 3.10 and CUDA 12.0.

To run this project, you'll need to install dependencies with 
```
pip install -r requirements.txt
```

You may also need to compile a cython extension to get faster replay buffer sampling efficiency:
```
./compile.sh
```

## Quickstart


To run the basic comparison with the four-room task, you can run the following command:

```bash
seed_offset=10 # offset for the random seed
nt=5           # number of trials


extra=Demo
analyze=False
debug=False

# first run 11
lr=5e-4                          # learning rate
d_kl_c=0.05                      # kl divergence coefficient
bs=256                           # batch size
agv=11                           # agent view size
size=11                          # environment size
shape=${size}x${size}            # environment shape
maxsteps=140                     # max steps per an agent can execute in a single episode
task=tasks.TVMGFR-$shape-RARG-GI # task name
qh_dim=128                       # qnet hidden dimension
epochs=40

if [ "$debug" = "True" ]; then
    train_n=2
    test_n=2
else
    train_n=10
    test_n=100
fi

# ===== classical DQL, no subgoal =====
for ((i = 1; i <= $nt; i++)); do
    content=$(url_encode "$extra DQL ($i/$nt) @$machine_name started")
    python run_gridworld.py \
        --seed $((i + seed_offset)) \
        train \
        -e $task \
        --policy DQL \
        --agent-view-size $agv \
        --max-steps $maxsteps \
        --extra $extra \
        --epochs 15 \
        --lr $lr \
        --d-kl-c $d_kl_c \
        --batch-size $bs \
        --subgoal-on False \
        --planning True \
        --qh-dim $qh_dim \
        --analyze $analyze \
        --sampling-strategy 4 \
        --debug $debug || exit 1
done

# ===== GOLSAv2 w RL: GR+DQL =====
for ((i = 1; i <= $nt; i++)); do
    content=$(url_encode "$extra GR w/ RL ($i/$nt) @$machine_name started")
    python run_gridworld.py \
        --seed $((i + seed_offset)) \
        train \
        -e $task \
        --policy DQLG \
        --agent-view-size $agv \
        --max-steps $maxsteps \
        --extra $extra \
        --epochs 10 \
        --lr $lr \
        --d-kl-c $d_kl_c \
        --batch-size $bs \
        --subgoal-on True \
        --planning True \
        --qh-dim $qh_dim \
        --analyze $analyze \
        --sampling-strategy 4 \
        --debug $debug || exit 1

done
```
Other experiments follow a similar pattern, see the corresponding flags in `run_gridworld.py`, `run_robot_arm.py`, `run_treasure_hunting.py` and `run_fmri.py`.

