# Nemotron-3-Nano-4B with 8xH100/H200

A dense 4B Mamba+Attention hybrid (`nemotron_h`) using
`megatron.bridge`'s AutoBridge path (no `torch_dist` conversion step —
weights are loaded straight from HF safetensors at run time).

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
# hf checkpoint
hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --local-dir /root/NVIDIA-Nemotron-3-Nano-4B-BF16

# train data
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
```

No `torch_dist` conversion is needed. The AutoBridge (`--megatron-to-hf-mode bridge`)
reads the HF safetensors directly the first time the actor initializes.

## Run Training

Execute the training script:

```bash
cd /root/miles
bash scripts/run-nemotron-3-nano-4b.sh
```

### Parameter Introduction

The script [scripts/run-nemotron-3-nano-4b.sh](https://github.com/radixark/miles/blob/main/scripts/run-nemotron-3-nano-4b.sh)
follows the same layout as the Qwen3 examples. A few model-specific knobs:

1. Architecture args come from [scripts/models/nemotron-3-nano-4b.sh](https://github.com/radixark/miles/blob/main/scripts/models/nemotron-3-nano-4b.sh):

   ```bash
   MODEL_ARGS=(
      --num-layers 42
      --hidden-size 3136
      --num-attention-heads 40 --num-query-groups 8 --kv-channels 128
      --ffn-hidden-size 12544
      --normalization RMSNorm --position-embedding-type none
      --vocab-size 131072 --make-vocab-size-divisible-by 128
      --untie-embeddings-and-output-weights
   )
   ```

   The hybrid layer pattern and Mamba-side hyperparameters are provided
   by the upstream `NemotronHModelProvider4B` in `megatron.bridge`, so
   they don't need to appear in `MODEL_ARGS`.

2. The CKPT args use bridge mode (no `--load` / `--ref-load` to a
   `_torch_dist` directory):

   ```bash
   CKPT_ARGS=(
      --hf-checkpoint /root/NVIDIA-Nemotron-3-Nano-4B-BF16
      --ref-load /root/NVIDIA-Nemotron-3-Nano-4B-BF16
      --save /root/nemotron-3-nano-4b-ckpts
      --save-interval 20
      --megatron-to-hf-mode bridge
   )
   ```

3. Default parallelism is TP=2. The model also trains cleanly under
   TP=4, PP=2, CP=2, or TP=2×PP=2 — flip the numbers in `PERF_ARGS`
   (the script's defaults are just one of the passing cells). Because
   `nemotron_h` is a Mamba-hybrid there is no expert parallelism for the
   dense 4B variant.

### Notes

- No MTP / speculative head.
- If you hit `ValueError: ... MoE and tensor parallelism ... without
  sequence parallelism` when raising TP, add `--sequence-parallel` to
  `PERF_ARGS` (needed whenever TP>1 interacts with the shared expert
  norm path).

## Known-Passing Parallelism Cells

10-step RL smoke with `train_rollout_logprob_abs_diff < 0.02`:

| Cell       | max diff | notes |
|------------|----------|-------|
| TP=2       | 0.01086  |       |
| TP=4       | 0.01174  |       |
| PP=2       | 0.01176  | PP-unwrap shim required |
| CP=2       | 0.01188  |       |
| TP=2×PP=2  | 0.01171  | exercises both shims  |
