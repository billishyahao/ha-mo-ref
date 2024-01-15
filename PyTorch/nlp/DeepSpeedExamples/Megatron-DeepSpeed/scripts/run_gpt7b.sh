# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.

#!/bin/bash

#This is based on https://github.com/bigscience-workshop/bigscience/blob/master/train/tr1-13B-base/tr1-13B-round1.slurm

set -ex

# ----------------------
# Configurable parameters
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/oscar-en/}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-meg-gpt2_text_document}
NUM_NODES=${HL_NUM_NODES:-1}
DP=${HL_DP:-2}
TP=${HL_TP:-4}
PP=${HL_PP:-1}
MICRO_BATCH=${HL_MICRO_BATCH:-1}
GLOBAL_BATCH=${HL_GBS:-1024}
SEQ_LEN=${HL_SEQ_LEN:-2048}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH:-}
HOSTSFILE=${HL_HOSTSFILE:-}
USE_HPU=${HL_USE_HPU:-1}
CKP_ACT=${HL_CKP_ACT:-0}
RAMPUP_BS=${HL_RAMPUP_BS:-1}
UNIV_CP=${HL_UNIV_CP:-0}
QNPU_DIR=${HL_QNPU_DIR:-}
TRAIN_ITER=${HL_TRAIN_ITERS:-10}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
N_LAYERS=${HL_NUM_LAYERS:-32}
N_GPU_PER_NODE=${HL_NGPU_PER_NODE:-8}
ZERO_STAGE=${HL_ZERO_STAGE:-0}
PROFILE=${HL_PROFILE:-} #provide either of pt, pt-full, hltv such as HL_PROFILE=hltv
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-0} #set to 1 to enable sequence parallelism
OPTIMIZER=${HL_OPTIMIZER:-adamw}
USE_FUSED_SDPA=${HL_USE_FUSED_SDPA:-false}
USE_FUSED_SDPA_WITH_RECOMPUTE=${HL_USE_FUSED_SDPA_WITH_RECOMPUTE:-true}
FP8_ENABLE=${HL_FP8_ENABLE:-0}
# ----------------------

if [[ -z "$MODEL_REFERENCES_ROOT" ]]; then
    echo "Must provide MODEL_REFERENCES_ROOT in environment" 1>&2
    exit 1
fi

DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}
MODEL_DIR=$MODEL_REFERENCES_ROOT/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed

# Scaling
NGPU_PER_NODE=${N_GPU_PER_NODE}
NUM_GPUs=$(($DP * $TP * $PP))

# Bloom-13B model architecture
NLAYERS=${N_LAYERS} # must be divisible by PP
NHIDDEN=4096
NHEADS=32 # must be divisible by TP
FFN_HIDDEN_SIZE=$(($NHIDDEN * 4))

# Experiment name
if [ -z "$EXP_NAME" ]; then
    EXP_NAME="default"
fi

# output paths
if [ -z "$OUTPUT_DIR" ]; then
    RUNTIME=`date +"%Y%m%d_%H%M"`
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/bloom13b/ds_${EXP_NAME}_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_GPUs${NUM_GPUs}_${RUNTIME}
fi

if [ -z "$CHECKPOINTS_DIR" ]; then
    CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
fi

if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TENSORBOARD_DIR}

# handle kill switch argument
if [ -z "$KILL_SWITCH_FILE"]; then
    KILL_SWITCH_ARG=""
else
    KILL_SWITCH_ARG="--kill-switch-path $KILL_SWITCH_FILE"
fi

PARTITIONED_MODE="\"auto\""
if [ $SEQ_PARALLEL -eq 1 ]; then
    PARTITIONED_MODE="false"
fi

DS_CONFIG=ds_config.json

# debug
DS_COMMS_LOGGER=
if [ "${COMMS_LOGGER}" == "true" ]; then
    DS_COMMS_LOGGER="
        \"comms_logger\": {
        \"enabled\": true,
        \"verbose\": true,
        \"prof_all\": true,
        \"debug\": true
        },"
fi
DS_FLOPS_PROFILER=
if [ "${FLOPS_PROFILER}" == "true" ]; then
    DS_FLOPS_PROFILER="
        \"flops_profiler\": {
            \"enabled\": true,
            \"profile_step\": 1,
            \"module_depth\": -1,
            \"top_modules\": 1,
            \"detailed\": true,
            \"output_file\": null
        },"
fi
DATA_TYPE=
if [ ${ZERO_STAGE} -eq 1 ]; then
    DATA_TYPE="
      \"data_types\":{
        \"grad_accum_dtype\":\"fp32\"
      },
    "
fi

cat << EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": $LOG_INTERVAL,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true,
    "accumulate_grads_via_hooks": true
  },
  "fp16": {"enabled": false},
  "wall_clock_breakdown": true,
  "memory_breakdown": false,
  ${DATA_TYPE}
  "pipeline": {
    "pipe_partitioned": $PARTITIONED_MODE,
    "grad_partitioned": $PARTITIONED_MODE
  }
}
EOT

# configure multi-node
MULTINODE_CMD=""
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]; then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi

# training script command
CMD=""
if [ ! -z "$QNPU_DIR" ]; then
    CMD="source ${QNPU_DIR}/activate ;"
fi

if [ ${FP8_ENABLE} -eq 1 ]; then
    FP8_PARAMS="--use-hpu-fp8-transformer-engine --cache-fp8-weight --cache-fp8-weight-fwd true --flatten-linear-operands"
else
    FP8_PARAMS=""
fi

CMD="${CMD} \
    cd $MODEL_DIR && \
    python3 -u ./pretrain_gpt.py \
    --deepspeed \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --train-iters ${TRAIN_ITER} \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters 20 \
    --eval-interval 100 \
    --data-path ${DATA_PATH} \
    --data-impl mmap \
    --split 949,50,1 \
    --vocab-file $DATA_DIR/gpt2-vocab.json \
    --merge-file $DATA_DIR/gpt2-merges.txt \
    --optimizer ${OPTIMIZER} \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-6 \
    --lr 3e-4 \
    --lr-decay-style cosine \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --load $CHECKPOINTS_DIR \
    --deepspeed_config=$DS_CONFIG  \
    --zero-stage=$ZERO_STAGE \
    --seed 42 \
    --exit-interval $EXIT_INTERVAL \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --use-fused-sdpa $USE_FUSED_SDPA \
    --use-fused-sdpa-with-recompute $USE_FUSED_SDPA_WITH_RECOMPUTE \
    $KILL_SWITCH_ARG \
    --bf16 ${FP8_PARAMS}"

if [ $USE_HPU -eq 1 ]
then
    CMD="${CMD} --use_hpu --distributed-backend=hccl --hpu-deterministic"
fi

if [ $SEQ_PARALLEL -eq 1 ]
then
    CMD="${CMD} --sequence-parallel"
fi

if [ $UNIV_CP -eq 1 ]
then
    echo "Loading Universal Checkpoint from ${CHECKPOINTS_DIR}"
    CMD="${CMD} --universal-checkpoint"
fi

if [ $CHECKPOINT_SAVE -eq 1 ]
then
    mkdir -p ${CHECKPOINTS_DIR}
    CMD="${CMD} --save $CHECKPOINTS_DIR --save-interval $SAVE_INTERVAL --verify-checkpoint --verify-checkpoint-model-type BLOOM"
fi

if [ $CKP_ACT -eq 1 ]
then
    CMD="${CMD} --checkpoint-activations --deepspeed-activation-checkpointing"
elif [ $CKP_ACT -eq 2 ]
then
    CMD="${CMD} --checkpoint-activations --deepspeed-activation-checkpointing --checkpoint-activations-granularity selective"
fi

if [ ! -z "$PROFILE" ]; then
    CMD="${CMD} --profile ${PROFILE}"
fi

if [ ! -z "$QNPU_DIR" ]; then
    rm -rf $HOME/.deepspeed_env
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $HOME/.deepspeed_env
fi

# run!
deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          $MULTINODE_CMD \
          /usr/bin/bash -c "$CMD" #2>&1 | tee ${OUTPUT_DIR}/log.txt
