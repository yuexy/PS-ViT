#!/usr/bin/env bash

NOW="`date +%Y%m%d%H%M%S`"
PARTITION=$1
JOB_NAME=$2
CONFIG=$3
NUM_PROC=$4

if [ ! -d "output" ];then
    mkdir output
fi

srun --mpi=pmi2 -n${NUM_PROC} -p ${PARTITION} --gres=gpu:8 \
--ntasks-per-node=8 --cpus-per-task=5 --job-name=${JOB_NAME} \
python main.py --config ${CONFIG} \
2>&1 | tee output/${JOB_NAME}_${NOW}.log
