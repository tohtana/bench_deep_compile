PROFILE_DIR=${PROFILE_DIR:-profiles}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
N3Z_OPTS="--compile --schedule"
AC_OPTS="--activation-checkpointing"

MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
BATCH_SIZE_OPTS=(1)
SEQ_LENGTH_OPTS=(1024)
ACC_OPTS=(2 4 8 16)
for ACC_STEP in ${ACC_OPTS[@]}; do
    for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
        for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${AC_OPTS} ${PROFILE_OPTS} --gradient-accumulation-steps ${ACC_STEP}"
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes prefetch,selective_gather
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes prefetch
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes selective_gather
        cp -r logs ${PROFILE_DIR}/
        done
    done
done

MODEL="mistralai/Mixtral-8x7B-v0.1"
BATCH_SIZE_OPTS=(1)
SEQ_LENGTH_OPTS=(1024)
ACC_OPTS=(2 4 8 16)
for ACC_STEP in ${ACC_OPTS[@]}; do
    for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
        for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
            ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${AC_OPTS} ${PROFILE_OPTS} --gradient-accumulation-steps ${ACC_STEP}"
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes prefetch,selective_gather
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes prefetch
        NUM_NODES=4 NGPUS_PER_NODE=8 bash ./run_multinode.sh --backend deepspeed ${ARGS} ${N3Z_OPTS} --passes selective_gather
        cp -r logs ${PROFILE_DIR}/
        done
    done
done
