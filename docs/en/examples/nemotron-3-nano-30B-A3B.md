# Nemotron-3-Nano-30B-A3B with 8xH100/H200

A 30B-total / 3B-active Mamba+Attention+MoE hybrid (`nemotron_h`). MoE
routing follows DeepSeek-V3 style: sigmoid scoring with an aux-free
expert bias, `top_k=6` of 128 experts, one shared expert per layer,
`routed_scaling_factor=2.5`. This config uses `megatron.bridge`'s
AutoBridge path plus the miles bridge shim that extends it for MoE
nemotron_h variants.

## Environment Setup

After pulling the `radixark/miles:latest` image, initialize the image environment as follows:

```bash
cd /root/
git clone https://github.com/radixark/miles.git
cd miles/
pip install -e . --no-deps

# Mamba SSM kernel — required by the nemotron_h hybrid backbone.
pip install mamba-ssm causal-conv1d
```

Download the model and data:

```bash
# hf checkpoint (~59 GB bf16)
hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --local-dir /root/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# train data
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
```

No `torch_dist` conversion is needed. The AutoBridge reads the HF
safetensors directly. The miles shim at
[miles/backends/megatron_utils/\_\_init\_\_.py](https://github.com/radixark/miles/blob/main/miles/backends/megatron_utils/__init__.py)
pulls `routed_scaling_factor`, `n_group`, `topk_group` and the MoE
routing dtype out of the HF config and onto the Megatron provider —
without this the routed expert output is silently scaled 1.0× instead
of 2.5×, producing a ~0.28 train/rollout logprob drift at RL time.

## Run Training

Execute the training script:

```bash
cd /root/miles
bash scripts/run-nemotron-3-nano-30b-a3b.sh
```

### Parameter Introduction

The script [scripts/run-nemotron-3-nano-30b-a3b.sh](https://github.com/radixark/miles/blob/main/scripts/run-nemotron-3-nano-30b-a3b.sh)
is patterned on the GLM-4.5 / Qwen3-30B examples. The model-specific
knobs live in
[scripts/models/nemotron-3-nano-30b-a3b.sh](https://github.com/radixark/miles/blob/main/scripts/models/nemotron-3-nano-30b-a3b.sh):

1. **MoE routing** (standard Megatron flags):

   ```bash
   MODEL_ARGS=(
      ...
      --num-experts 128
      --moe-router-topk 6
      --moe-ffn-hidden-size 1856
      --moe-shared-expert-intermediate-size 3712
      --moe-router-score-function sigmoid
      --moe-router-enable-expert-bias
      --moe-grouped-gemm
      --moe-router-dtype fp32
      --moe-router-num-groups 1
      --moe-router-group-topk 1
      --moe-router-topk-scaling-factor 2.5
      --moe-router-pre-softmax
      --moe-router-load-balancing-type seq_aux_loss
      --moe-router-bias-update-rate 0
      --moe-aux-loss-coeff 0
   )
   ```

2. **Parallelism — bridge + MoE on 8xH200**. With the default
   `--rollout-max-response-len 1024` + `--n-samples-per-prompt 4` +
   `--max-tokens-per-gpu 1024` + `--log-probs-chunk-size 128`, the
   following cells have all passed the <0.02 bar:

   | Cell               | max diff | notes |
   |--------------------|----------|-------|
   | EP=4 pure          | 0.01438  |       |
   | TP=2×EP=4+SP       | 0.01431  |       |
   | PP=2×EP=4          | 0.01436  | PP-unwrap shim required |
   | CP=2×EP=4          | 0.01428  |       |
   | TP=2×PP=2×EP=2+SP  | 0.01436  | mixed 3-way sharding  |

   `TP=4×EP=2` is infeasible on 8xH200: 15.2B params per rank plus the
   sglang colocate copy pushes each GPU past its 140 GB budget. Use
   TP=2 or drop back to pure EP=4 if you must stay at this batch size.

3. **Rollout routing replay** — required for MoE:

   ```bash
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 1
      --sglang-mem-fraction-static 0.7
      --use-miles-router
      --use-rollout-routing-replay
   )
   ```

### Notes

- No MTP / speculative head.
- The model uses squared-ReLU activation in both dense shared experts
  and routed experts (non-gated MLP). The miles bridge shim maps the
  routed expert weights with `AutoMapping`, not `GatedMLPMapping`.
- Memory on 8xH200 is tight once sglang is colocated. If you need a
  longer response length than 1024, shard Megatron further (e.g.
  TP=2×PP=2) before raising the sequence cap.
