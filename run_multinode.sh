#!/bin/bash

NUM_NODES=${NUM_NODES:-$(wc -l < /job/hostfile)}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
NUM_PROCESSES=$((${NUM_NODES} * ${NGPUS_PER_NODE}))

BACKEND="deepspeed"
MODEL="meta-llama/Meta-Llama-3-8B"
ZERO_STAGE=3
COMPILE=0
PASSES="ALL"
EXTRA_OPTS=""

SCHEDULE=0
GRADIENT_ACCUMULATION_STEPS=1
ACTIVATION_CHECKPOINTING=1
BATCH_SIZE=1
SEQ_LENGTH=512


while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --zero-stage)
            ZERO_STAGE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --batch_size $2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --seq_length $2"
            shift 2
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            # EXTRA_OPTS="${EXTRA_OPTS} --gradient_accumulation_steps $2"
            shift 2
            ;;
        --activation-checkpointing)
            ACTIVATION_CHECKPOINTING=1
            EXTRA_OPTS="${EXTRA_OPTS} --activation_checkpointing"
            shift
            ;;   
        --compile)
            COMPILE=1
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
        --schedule)
            SCHEDULE=1
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
        --passes)
            PASSES="$2"
            EXTRA_OPTS="${EXTRA_OPTS} $1 $2"
            shift 2
            ;;
        --profile)
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
        --profile-dir)
            EXTRA_OPTS="${EXTRA_OPTS} --profile_dir $2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "NUM_NODES: ${NUM_NODES} NGPUS_PER_NODE: ${NGPUS_PER_NODE} NUM_PROCESSES: ${NUM_PROCESSES}"

HOST_IP=$(hostname -i)

mkdir -p logs

SCRIPT_DIR=$(dirname $(realpath $0))

#replace , with _ in PASSES
PASSES=$(echo $PASSES | tr ',' '_')

ds_ssh -f hostfile_n${NUM_NODES} "cd ${SCRIPT_DIR}; BLOB_BASE_DIR=${BLOB_BASE_DIR} bash ./run.sh ${HOST_IP} ${NUM_NODES} ${NUM_PROCESSES} ${BACKEND} ${ZERO_STAGE} ${MODEL} ${GRADIENT_ACCUMULATION_STEPS} ${EXTRA_OPTS} \
    2>&1 | tee logs/debug_${MODEL#*/}_${BACKEND}_np${NUM_PROCESSES}c${COMPILE}s${SCHEDULE}b${BATCH_SIZE}seq${SEQ_LENGTH}g${GRADIENT_ACCUMULATION_STEPS}a${ACTIVATION_CHECKPOINTING}p${PASSES}.log"
