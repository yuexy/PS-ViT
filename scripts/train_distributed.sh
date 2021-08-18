#!/usr/bin/env bash

NOW="`date +%Y%m%d%H%M%S`"
JOB_NAME=$1
CONFIG=$2
NUM_PROC=$3
MASTER_PORT=2333

if [ ! -d "output" ];then
    mkdir output
fi

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=${MASTER_PORT} main.py --config=${CONFIG} --distributed=True 2>&1 | tee output/${JOB_NAME}_${NOW}.log
