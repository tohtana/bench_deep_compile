import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def throughput_calculator(micro_batch_size, acc_steps, np, elapsed_time_per_iter,
                          hidden_size, num_attention_heads, num_key_value_heads,
                          ffn_hidden_size, num_layers, padded_vocab_size, seq_len,
                          topk: int, swiglu: bool, checkpoint_activations: bool):
    batch_size = micro_batch_size * acc_steps * np
    samples_per_second = batch_size / elapsed_time_per_iter

    head_dim = hidden_size // num_attention_heads
    gqa = num_attention_heads // num_key_value_heads
    ffn_multiplier = 3 if swiglu else 2
    macs_per_flops = 2

    pre_and_post_mha_gemm_macs = batch_size * num_layers * (1 + (2 // gqa) + 1) * (hidden_size**2) * seq_len
    mha_bgemm_macs = batch_size * num_layers * 2 * head_dim * num_attention_heads * (seq_len**2)
    ffn_gemm_macs = batch_size * num_layers * ffn_multiplier * ffn_hidden_size * hidden_size * seq_len * topk
    logit_lmhead_gemm_macs = batch_size * padded_vocab_size * hidden_size * seq_len

    fwd_macs = pre_and_post_mha_gemm_macs + mha_bgemm_macs + ffn_gemm_macs + logit_lmhead_gemm_macs
    bwd_macs = 2 * fwd_macs
    fwd_bwd_macs = fwd_macs + bwd_macs

    if checkpoint_activations:
        fwd_bwd_macs += fwd_macs

    flops_per_iteration = fwd_bwd_macs * macs_per_flops
    tflops = flops_per_iteration / (elapsed_time_per_iter * np * (10**12))
    return samples_per_second, tflops


model_info = {
    "meta-llama/Meta-Llama-3-8B": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "ffn_hidden_size": 16384,
        "num_layers": 32,
        "padded_vocab_size": 32000,
        "topk": 1,
        "swiglu": True
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "ffn_hidden_size": 32768,
        "num_layers": 80,
        "padded_vocab_size": 32000,
        "topk": 1,
        "swiglu": True
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "ffn_hidden_size": 16384,
        "num_layers": 32,
        "padded_vocab_size": 32000,
        "topk": 2,
        "swiglu": False
    }
}

parser = argparse.ArgumentParser(description="Plot performance metrics.")
parser.add_argument("--metric", choices=["iteration_time", "throughput", "flops", "mfu", "peak_mem"], required=True,
                    help="Metric to plot: 'iteration_time', 'flops', 'mfu', or 'peak_mem'")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the directory containing results.txt")
parser.add_argument("--result_file", type=str, default="results.txt", help="Name of the result file")
parser.add_argument("--acc_step_eval", action="store_true", help="Evaluate the accuracy of the model")
args = parser.parse_args()


pattern = re.compile(
    r"(?P<timestamp>\d+) (?P<model>[\w./-]+) ds=(?P<ds>\w+) np=(?P<np>\d+) batch_size=(?P<batch_size>\d+) "
    r"seq=(?P<seq>\d+) zero_stage=(?P<zero_stage>\d+) acc=(?P<acc>\d+) ac=(?P<ac>\w+) compile=(?P<compile>\w+) schedule=(?P<schedule>\w+) "
    f"passes=(?P<passes>[\w,_]+) compile_time=(?P<compile_time>[\d.]+) iteration time: (?P<iteration_time>[\d.]+) "
    r"alloc_mem: (?P<alloc_mem>\d+) peak_mem: (?P<peak_mem>\d+)"
)
file = Path(args.result_dir) / args.result_file
matches = []
with open(file) as f:
    for line in f:
        match = pattern.match(line)
        if not match:
            print(f"Not matched: {line}")
        if match:
            d = match.groupdict()
            if "passes" not in d:
                d["passes"] = ""
            if "compile_time" not in d:
                d["compile_time"] = 0
            if "schedule" not in d:
                d["schedule"] = d["compile"]
            matches.append(d)

df = pd.DataFrame(matches)
print(df)

df["ds"] = df["ds"] == "True"
df["compile"] = df["compile"] == "True"
df["np"] = df["np"].astype(int)
df["batch_size"] = df["batch_size"].astype(int)
df["seq"] = df["seq"].astype(int)
df["iteration_time"] = df["iteration_time"].astype(float)
df["alloc_mem"] = df["alloc_mem"].astype(float)
df["peak_mem"] = df["peak_mem"].astype(float)
df["acc"] = df["acc"].astype(int)
df["ac"] = df["ac"] == "True"
df["compile_time"] = df["compile_time"].astype(float)
df["schedule"] = df["schedule"] == "True"


grouped = df.groupby(["model", "np", "batch_size"])

# We used A100
theoretical_peak = 312

LABEL_ZERO3 = "ZeRO3"
LABEL_ZERO3_C = "ZeRO3 (C)"
LABEL_FSDP = "FSDP"
LABEL_FSDP_C = "FSDP (C)"
LABEL_DC_PS = "DeepCompile (P+S)"
LABEL_DC_P = "DeepCompile (P)"
LABEL_DC_S = "DeepCompile (S)"

for (model, np, batch_size), group in grouped:

    sort_group_name = "acc" if args.acc_step_eval else "seq"

    group = group.sort_values(sort_group_name)
    labels = group[sort_group_name].unique()

    metric_values = {LABEL_ZERO3: [0] * len(labels),
                    LABEL_ZERO3_C: [0] * len(labels),
                    LABEL_FSDP: [0] * len(labels),
                    LABEL_FSDP_C: [0] * len(labels),
                    LABEL_DC_PS: [0] * len(labels),
                    LABEL_DC_P: [0] * len(labels),
                    LABEL_DC_S: [0] * len(labels)}
                    
    for _, row in group.iterrows():
        if row["ds"] and not row["compile"]:
            category = LABEL_ZERO3
        elif not row["ds"]:
            if row["compile"]:
                category = LABEL_FSDP_C
            else:
                category = LABEL_FSDP
        elif row["ds"] and row["compile"]:
            if not row["schedule"]:
                category = LABEL_ZERO3_C
            elif row["passes"] == "" or row["passes"] == 'prefetch,selective_gather':
                category = LABEL_DC_PS
            elif row["passes"] == 'prefetch':
                category = LABEL_DC_P
            elif row["passes"] == 'selective_gather':
                category = LABEL_DC_S
            else:
                print(f"Unknown category1 : {row}")
                continue
        else:
            print(f"Unknown category2 : {row}")
            continue

        group_index = list(labels).index(row[sort_group_name])
        if args.metric == "iteration_time":
            metric_values[category][group_index] = row["iteration_time"]
        elif args.metric == "peak_mem":
            metric_values[category][group_index] = row["peak_mem"] / (1024**3)
        elif args.metric == "throughput":
            metric_values[category][group_index] = row["batch_size"] * row["seq"] / row["iteration_time"] * row["acc"]
        elif args.metric in ["flops", "mfu"]:
            model_params = model_info[row["model"]]
            samples_per_second, tflops = throughput_calculator(
                micro_batch_size=row["batch_size"],
                acc_steps=row["acc"],
                np=row["np"],
                elapsed_time_per_iter=row["iteration_time"],
                hidden_size=model_params["hidden_size"],
                num_attention_heads=model_params["num_attention_heads"],
                num_key_value_heads=model_params["num_key_value_heads"],
                ffn_hidden_size=model_params["ffn_hidden_size"],
                num_layers=model_params["num_layers"],
                padded_vocab_size=model_params["padded_vocab_size"],
                seq_len=row["seq"],
                topk=model_params["topk"],
                swiglu=model_params["swiglu"],
                checkpoint_activations=row["ac"]
            )
            if args.metric == "flops":
                metric_values[category][group_index] = tflops
            elif args.metric == "mfu":
                metric_values[category][group_index] = tflops / theoretical_peak

    x = range(len(labels))
    width = 0.1
    ylabel = {
        "iteration_time": "Iteration Time (s)",
        "flops": "TFLOPS",
        "throughput": "Throughput (tokens/s/GPU)",
        "mfu": "MFU",
        "peak_mem": "Peak Memory (GB)"
    }[args.metric]

    if args.metric == "peak_mem":
        plt.figure(figsize=(7, 8))
    else:
        plt.figure(figsize=(10, 8))
    adjust = - .0 * width
    plt.bar([i - width*3 + adjust for i in x], metric_values[LABEL_ZERO3], width, label=LABEL_ZERO3, alpha=0.7)
    plt.bar([i - width*2 + adjust for i in x], metric_values[LABEL_ZERO3_C], width, label=LABEL_ZERO3_C, alpha=0.7)
    plt.bar([i - width + adjust for i in x], metric_values[LABEL_FSDP], width, label=LABEL_FSDP, alpha=0.7)
    plt.bar([i + adjust for i in x], metric_values[LABEL_FSDP_C], width, label=LABEL_FSDP_C, alpha=0.7)
    plt.bar([i + width + adjust for i in x], metric_values[LABEL_DC_P], width, label=LABEL_DC_P, alpha=0.7)
    plt.bar([i + width*2 + adjust for i in x], metric_values[LABEL_DC_S], width, label=LABEL_DC_S, alpha=0.7)
    plt.bar([i + width*3 + adjust for i in x], metric_values[LABEL_DC_PS], width, label=LABEL_DC_PS, alpha=0.7)

    gain_zero3 = [metric_values[LABEL_DC_PS][i] / metric_values[LABEL_ZERO3][i] for i in range(len(labels))]
    print(f"model {model} np {np} batch_size {batch_size} {LABEL_ZERO3} metric_values: {metric_values[LABEL_ZERO3]} gain_zero3: {gain_zero3}")
    gain_fsdp = [0 if metric_values[LABEL_FSDP][i] == 0 else metric_values[LABEL_DC_PS][i] / metric_values[LABEL_FSDP][i] for i in range(len(labels))]
    print(f"model {model} np {np} batch_size {batch_size} {LABEL_FSDP} metric_values: {metric_values[LABEL_FSDP]} gain_fsdp: {gain_fsdp}")
    print(f"model {model} np {np} batch_size {batch_size} {LABEL_DC_PS} metric_values: {metric_values[LABEL_DC_PS]}")

    model = model.split('/')[1]
    model = model.replace("Meta-Llama-3-8B", "Llama-3-8B")
    model = model.replace("Meta-Llama-3-70B-Instruct", "Llama-3-70B")
    model = model.replace("Mixtral-8x7B-v0.1", "Mixtral-8x7B")

    plt.title(f"{model}, #GPUs: {np}, Batch Size: {batch_size}", fontsize=20)
    if args.acc_step_eval:
        plt.xlabel("Accumulation Steps", fontsize=20)
    else:
        plt.xlabel("Sequence Length", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(x, labels, fontsize=20)
    plt.yticks(fontsize=20)

    if args.metric == "peak_mem":
        plt.ylim(0, 80)
        plt.legend(loc="lower right", fontsize=16)
    else:
        plt.legend(loc="lower right", fontsize=18)

    plt.grid(axis="y")

    metric_name = args.metric
    model = model.replace("/", "_")
    chart_dir = Path(args.result_dir) / Path(metric_name)
    chart_dir.mkdir(parents=True, exist_ok=True)
    conf_str = f"{metric_name}_{model}_np{np}_bs{batch_size}"
    img_path = chart_dir / f"chart_{conf_str}.png"
    plt.savefig(str(img_path))
    plt.close()
