#!/bin/bash

set -x

MY_SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"

export MODEL_REFERENCES_ROOT="${MY_SCRIPT_DIR}/../../../.."

# do not show me git error
git config --global --add safe.directory '*'

# dirs
export HL_DATA_DIR_ROOT="/data/oscar-en/"
export HL_HOSTSFILE="${MY_SCRIPT_DIR}/scripts/hostsfile"

# model setting
export HL_NUM_LAYERS=32
export HL_SEQ_LEN=2048

# 13B Single Node
export HL_NUM_NODES=1
export HL_TP=1
export HL_PP=2
export HL_DP=4


# SP
#export HL_SEQ_PARALLEL=1

# FusedSDPA
export HL_USE_FUSED_SDPA=true
export HL_USE_FUSED_SDPA_WITH_RECOMPUTE=true

# batch
export HL_MICRO_BATCH=1
export HL_GBS=1024

# ZeRO
export HL_ZERO_STAGE=1

# fp8
export HL_FP8_ENABLE=0

# Checkpoint
export HL_CKP_ACT=0
#export HL_CKP_ACT=2

# iterations to run
#export HL_TRAIN_ITERS=10000
export HL_TRAIN_ITERS=200
export HL_EVAL_ITERS=0

# not save checkpoint
#export HL_SAVE=1
export HL_SAVE=0

# others
export HL_LOG_INTERVAL=1
export HL_KILL_SWITCH="${MY_SCRIPT_DIR}/kill-switch"


# Habana logs
#export HABANA_LOGS=${MY_SCRIPT_DIR}/habana_logs
#export LOG_LEVEL_ALL_HCL=0

bash ${MY_SCRIPT_DIR}/scripts/run_gpt7b.sh
