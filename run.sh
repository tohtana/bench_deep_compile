HOST_IP=$1
NUM_NODES=$2
NUM_PROCESSES=$3
BACKEND=$4
ZERO_STAGE=$5
MODEL=$6
GRADIENT_ACCUMULATION_STEPS=$7
shift 7
EXTRA_OPTS="$@"

export NCCL_DEBUG=WARN

CONFIG_TEMPLATE=configs/ds_config.yaml.template
if [ "${BACKEND}" == "fsdp" ]; then
    CONFIG_TEMPLATE=configs/fsdp_config.yaml.template
elif [ "${BACKEND}" == "ddp" ]; then
    CONFIG_TEMPLATE=configs/ddp_config.yaml.template
elif [ "${BACKEND}" != "deepspeed" ]; then
    echo "Invalid backend: ${BACKEND}"
    exit 1
fi

echo "HOST_IP: ${HOST_IP}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "BACKEND: ${BACKEND}"
echo "ZERO_STAGE: ${ZERO_STAGE}"
echo "MODEL: ${MODEL}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "EXTRA_OPTS: ${EXTRA_OPTS}"

MACHINE_RANK=$(hostname | sed 's/[^0-9]*//g')

python generate_conf.py \
    --machine_rank ${MACHINE_RANK} \
    --num_machines ${NUM_NODES} \
    --num_processes ${NUM_PROCESSES} \
    --zero_stage ${ZERO_STAGE} \
    --template_file ${CONFIG_TEMPLATE} \
    --output_file configs/config.yaml

GAS_OPTS="--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"
if [ "${BACKEND}" == "deepspeed" ]; then
    python generate_conf.py \
        --machine_rank ${MACHINE_RANK} \
        --num_machines ${NUM_NODES} \
        --num_processes ${NUM_PROCESSES} \
        --zero_stage ${ZERO_STAGE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --template_file configs/ds_config.json.template \
        --output_file configs/ds_config.json
fi

${HOME}/.local/bin/accelerate launch --main_process_ip ${HOST_IP} --main_process_port 12345 \
--num_machines ${NUM_NODES} --num_processes ${NUM_PROCESSES} --machine_rank ${MACHINE_RANK} \
--config_file configs/config.yaml \
run_acc_lm.py \
--model_name "${MODEL}" \
--zero_stage ${ZERO_STAGE} \
${GAS_OPTS} \
${EXTRA_OPTS}