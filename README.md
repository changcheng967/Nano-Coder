# Nano Coder

**Goal: Build the world's best 8B coding agent through knowledge distillation**

Training competitive coding models on Ascend 910ProA NPUs via MindFormers.

## Status

**NC-1 Preview**: Pipeline validation on OpenI platform

| Component | Status |
|-----------|--------|
| MindFormers config generation | Done |
| C2NET data loading | Done |
| 4-NPU pipeline parallelism | Done |
| FlashAttention patch (910ProA) | Done |
| GQA support | Done |

## Quick Start

```bash
# On OpenI platform with c2net context
python train.py
```

The script automatically:
1. Loads model/data paths from c2net context
2. Generates MindFormers YAML config
3. Patches FlashAttention for 910ProA compatibility
4. Launches 4-NPU distributed training with `msrun`

## Hardware

| Requirement | Value |
|-------------|-------|
| Platform | OpenI (openi.pcl.ac.cn) |
| NPUs | 4x Ascend 910ProA (32GB each) |
| Total Memory | 128GB HBM |
| Framework | MindFormers (MindSpore 2.5+) |

## Architecture

```
Training Pipeline (train.py)
├── C2NET Context
│   ├── Model path resolution
│   ├── Dataset path resolution
│   └── Output directory
├── Config Generation
│   ├── MindFormers YAML (finetune_qwen3_8b_lora.yaml)
│   └── Parallel speed-up JSON
├── FlashAttention Patch
│   ├── Backup original
│   └── Replace with BMM-Softmax-BMM
└── Distributed Training
    └── msrun (4 workers, pipeline parallel)
```

## Configuration

Key training settings in `train.py`:

```yaml
# Parallelism
model_parallel: 1        # mp=1 avoids Tile sharding bug
pipeline_stage: 4        # pp=4 uses all 4 NPUs
micro_batch_num: 4       # Required for pp=4

# Model
base_model: Qwen3-8B
lora_rank: 8
seq_length: 4096

# FlashAttention (910ProA unsupported)
use_flash_attention: False
MS_ENABLE_FLASH_ATTENTION: 0
MS_DEV_GRAPH_KERNEL_FLAGS: --disable_pass=FlashAttentionFusionV1,V2
```

## FlashAttention Patch

Ascend 910ProA lacks the FlashAttentionScore CANN kernel. The training script patches `flash_attention.py` at runtime to use standard ops:

```
FlashAttentionScore (CANN kernel) → BMM-Softmax-BMM (standard ops)
```

Features:
- Grouped Query Attention (GQA) support
- All-static reshape dimensions for graph mode
- Pipeline parallelism compatible

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed analysis.

| Stage | Model | Data | Target |
|-------|-------|------|--------|
| NC-1 Preview | 8B + LoRA | SWE-Lego subset | Pipeline validation |
| NC-1 | 8B + LoRA | CoderForge 155K | 40-48% |
| NC-2 | 8B + Scaffold | + verify-on-edit | 55-62% |
| NC-3 | 8B + RL | GRPO++ | 52-58% pass@1 |
| NC-5 | 32B + QLoRA | Full dataset | 65-75% |

## Key Fixes Applied

1. **FlashAttention disabled** - Env vars + config.json + YAML + monkey-patch
2. **Compiler passes disabled** - `MS_DEV_GRAPH_KERNEL_FLAGS` prevents FA fusion
3. **FlashAttention patched** - BMM-Softmax-BMM replacement with GQA support
4. **Pipeline parallelism** - mp=1, pp=4 to avoid Tile sharding bug
5. **Static reshape dims** - Graph-mode compatible with `self.head_num`, `self.head_dim`

## Folder Structure

```
Nano-Coder/
├── train.py                    # Main training script
├── README.md                   # This file
├── ROADMAP.md                  # Detailed roadmap with sources
└── swe_bench_verified_500.json # SWE-bench dataset
```

## References

- [MindFormers](https://gitee.com/mindspore/mindformers) - MindSpore training framework
- [CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) - 155K coding trajectories
- [SWE-bench Verified](https://www.swebench.com/) - Benchmark of 500 GitHub issues
- [OpenI Platform](https://openi.pcl.ac.cn) - Free Ascend NPU compute
