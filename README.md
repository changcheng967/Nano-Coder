# Nano Coder Series

**Goal: Build competitive open-source coding models through knowledge distillation**

Training specialized coding models from frontier teachers (Kimi K2.5 at 76.8% SWE-bench).

## Quick Start

```bash
# Set your NVIDIA NIM API key (free at build.nvidia.com)
export NVIDIA_API_KEY="nvapi-xxxxx"

# Run Stage 1 Preview (2 hours)
python train.py --max-tasks 100
```

## Research-Based Roadmap

Based on published results from SWE-Lego (Huawei), SWE-Star, and Devstral:

| Stage | Time | Model | Data | Expected |
|-------|------|-------|------|----------|
| Baseline | 0h | Qwen3-8B | - | ~25% |
| **NC1 Preview** | 2h | 8B + LoRA | 100-500 traj | 28-35% |
| **NC1 Full** | 24h | 8B + LoRA | 2k-5k traj | 35-42% |
| **NC2** | 72h | 32B + LoRA | 10k-20k traj | 45-52% |
| **NC3** | 1 week | 32B + Full FT | 20k+ traj | 50-57% |

See [ROADMAP.md](ROADMAP.md) for detailed analysis with sources.

## Ground Truth from Published Research

| Model | Pass@1 | TTS@16 | Source |
|-------|--------|--------|--------|
| Qwen3-8B baseline | ~25% | - | Raw model |
| SWE-Lego-Qwen3-8B | 42.2% | 49.6% | Huawei, Jan 2026 |
| SWE-Lego-Qwen3-32B | 52.6% | 58.8% | Huawei, Jan 2026 |
| SWE-Star-32B | 57.1% | - | 250k trajectories |
| Devstral (Mistral) | 46.8% | - | Prior SOTA |

**Key insight**: SWE-Lego achieved 42% with 18k trajectories. We're starting with 100-500.

## Architecture

```
Teacher Models (NVIDIA NIM)
├── Kimi K2.5 (76.8% SWE-bench) ← Primary teacher
├── DeepSeek V3.2 (73.0%)
└── Qwen3-Coder-480B (70.6%)

Distillation Pipeline
├── Generate trajectories from teachers
├── Filter by test verification
└── Fine-tune student with LoRA

Student Models
├── Stage 1-2: Qwen3-8B (16GB)
└── Stage 3+: Qwen3-32B or Qwen3-Coder-30B-A3B
```

## The Hard Truth

**8B models max out at ~42-50%** (proven by SWE-Lego with 18k trajectories).

**32B models max out at ~52-58%** (proven by SWE-Lego, SWE-Star).

To get 60%+ requires: 70B+ model OR better teacher OR reinforcement learning.

## Folder Structure

```
nanocoder/
├── train.py       # Unified training script
├── run.sh         # Quick start
├── README.md      # This file
└── ROADMAP.md     # Detailed roadmap with sources
```

## Hardware Requirements

- **Stage 1-2**: 4x Ascend 910 NPUs (32GB each) OR 1x A100 40GB
- **Stage 3+**: 4x A100 80GB or equivalent

## Key Techniques

1. **Knowledge Distillation**: Learn from 76.8% teacher
2. **Agentic Trajectories**: OpenHands tool-calling sequences
3. **Test Verification**: Only keep passing solutions
4. **LoRA Fine-tuning**: Efficient adaptation
5. **Test-Time Scaling**: +7-8% with verifier (proven by SWE-Lego)

## References

- [SWE-bench Verified](https://www.swebench.com/) - Benchmark of 500 GitHub issues
- [SWE-Lego](https://swe-lego.github.io/) - Huawei's SOTA 8B/32B results
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Training framework
- [NVIDIA NIM](https://build.nvidia.com) - Free API for frontier models
