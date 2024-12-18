PROFILE_DIR=${PROFILE_DIR:-profiles}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
N3Z_OPTS="--compile --schedule"
ACC_OPTS="--gradient-accumulation-steps 1"
AC_OPTS="--activation-checkpointing"


MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
BATCH_SIZE_OPTS=(1 2 4)
SEQ_LENGTH_OPTS=(512 1024 2048)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        # skip if batch size is 4 and seq length is 2048, as it causes OOM
        if [ ${BATCH_SIZE} -eq 4 ] && [ ${SEQ_LENGTH} -eq 2048 ]; then
            continue
        fi

        ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${ACC_OPTS} ${AC_OPTS} ${PROFILE_OPTS}"
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend fsdp ${ARGS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend fsdp ${ARGS} ${COMPILE_OPTS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes prefetch,selective_gather
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes prefetch
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes selective_gather

        cp -r logs ${PROFILE_DIR}/
    done
done

MODEL="mistralai/Mixtral-8x7B-v0.1"
BATCH_SIZE_OPTS=(1 2 4)
SEQ_LENGTH_OPTS=(512 1024 2048)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        # skip if batch size is 4 and seq length is 2048, as it causes OOM
        ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${ACC_OPTS} ${AC_OPTS} ${PROFILE_OPTS}"
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend fsdp ${ARGS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend fsdp ${ARGS} ${COMPILE_OPTS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes prefetch,selective_gather
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes prefetch
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes selective_gather

        cp -r logs ${PROFILE_DIR}/
    done
done
