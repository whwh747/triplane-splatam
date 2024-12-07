#!/bin/bash

for seed in 0
do
    SEED=${seed}
    export SEED
    for scene in 1 2 3 4
    do
        SCENE_NUM=${scene}
        export SCENE_NUM
        echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
        python3 -u scripts/splatam.py configs/tum/splatam.py
    done
done