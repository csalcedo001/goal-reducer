#!/bin/bash

seed_offset=10 # offset for the random seed
nt=2           # number of trials


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

# ===== classical DQL, no subgoal =====
for ((i = 1; i <= $nt; i++)); do
    # content=$(url_encode "$extra DQL ($i/$nt) @$machine_name started")
    python run_gridworld.py \
        --seed $((i + seed_offset)) \
        train \
        -e $task \
        --policy DQL \
        --agent-view-size $agv \
        --max-steps $maxsteps \
        --extra $extra \
        --epochs $epochs \
        --lr $lr \
        --d-kl-c $d_kl_c \
        --batch-size $bs \
        --subgoal-on False \
        --planning True \
        --qh-dim $qh_dim \
        --analyze $analyze \
        --sampling-strategy 4 \
        --debug $debug
done

# ===== GOLSAv2 w RL: GR+DQL =====
for ((i = 1; i <= $nt; i++)); do
    # content=$(url_encode "$extra GR w/ RL ($i/$nt) @$machine_name started")
    python run_gridworld.py \
        --seed $((i + seed_offset)) \
        train \
        -e $task \
        --policy DQLG \
        --agent-view-size $agv \
        --max-steps $maxsteps \
        --extra $extra \
        --epochs $epochs \
        --lr $lr \
        --d-kl-c $d_kl_c \
        --batch-size $bs \
        --subgoal-on True \
        --planning True \
        --qh-dim $qh_dim \
        --analyze $analyze \
        --sampling-strategy 4 \
        --debug $debug

done