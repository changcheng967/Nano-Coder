#!/usr/bin/env python3
"""
Nano Coder - OpenI Training Script (MindFormers)
=================================================
Fine-tune Qwen3-8B on SWE-Lego trajectories using MindFormers on Ascend NPU.

Hardware: 4× Ascend 910ProA (32 GB HBM each)
Platform: OpenI (c2net context for paths and upload)

Usage on OpenI:
    python train.py

The script expects:
    - Model: Pre-uploaded to OpenI model repo (c2net context)
    - Dataset: Pre-uploaded SWE-Lego parquet file (c2net context)
    - MindFormers: Installed via pip
"""

import os
import json
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime

# Ascend NPU settings
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0,1,2,3')
os.environ.setdefault('HCCL_CONNECT_TIMEOUT', '7200')
os.environ.setdefault('HCCL_EXEC_TIMEOUT', '7200')
os.environ.setdefault('MS_ENABLE_LCCL', '1')  # Lightweight collective comm for better 4-card performance
os.environ.setdefault('MS_BUILD_PROCESS_NUM', '32')  # Parallel graph compilation (192 cores / 4 workers)
os.environ.setdefault('MS_COMPILER_CACHE_ENABLE', '1')  # Cache compiled graphs for faster restarts
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # Avoid fork warnings
os.environ.setdefault('GLOG_v', '1')  # Show actual errors during compilation
os.environ['MS_ENABLE_FLASH_ATTENTION'] = '1'  # Enable native FlashAttention (works on 910ProA + CANN 8.3)
# os.environ['MS_ENABLE_FA_FLATTEN'] = 'off'  # Let CANN handle FA optimization
# os.environ['MS_DEV_GRAPH_KERNEL_FLAGS'] = '--disable_pass=FlashAttentionFusionV1,FlashAttentionFusionV2'  # FA fusion enabled


def find_executable(name: str) -> str:
    """Find an executable in PATH."""
    path = shutil.which(name)
    if path:
        return path
    raise RuntimeError(f"'{name}' not found in PATH. Ensure it's installed and accessible.")


# ============================================================================
# C2NET OPENI CONTEXT
# ============================================================================

def init_c2net():
    """Initialize c2net context for OpenI platform."""
    try:
        from c2net.context import prepare
        context = prepare()
        return context
    except ImportError:
        raise RuntimeError("c2net not installed. This script must run on OpenI platform.")
    except Exception as e:
        raise RuntimeError(f"c2net initialization failed: {e}")


def get_model_path(context) -> Path:
    """Get model path from c2net context."""
    if not hasattr(context, 'pretrain_model_path'):
        raise RuntimeError("Model not found in c2net context. Please upload model to OpenI repo.")

    model_dir = Path(context.pretrain_model_path)

    # Check if config.json exists directly
    if (model_dir / 'config.json').exists():
        return model_dir

    # Find subdirectory containing config.json (OpenI stores models in subdirs)
    for subdir in model_dir.iterdir():
        if subdir.is_dir() and (subdir / 'config.json').exists():
            return subdir

    raise RuntimeError(f"No model found in {model_dir}. Expected config.json in model directory or subdirectory.")


def get_dataset_path(context) -> Path:
    """Get dataset path from c2net context and find parquet file."""
    if not hasattr(context, 'dataset_path'):
        raise RuntimeError("Dataset not found in c2net context. Please upload dataset to OpenI repo.")

    dataset_dir = Path(context.dataset_path)

    # Find parquet file in dataset directory (recursive for nested structures)
    parquet_files = list(dataset_dir.rglob('*.parquet'))
    if parquet_files:
        return parquet_files[0]

    # Fallback to jsonl/json
    jsonl_files = list(dataset_dir.rglob('*.jsonl'))
    if jsonl_files:
        return jsonl_files[0]

    json_files = list(dataset_dir.rglob('*.json'))
    if json_files:
        return json_files[0]

    raise RuntimeError(f"No data files found in dataset directory: {dataset_dir}")


def upload_results(context):
    """Upload output results back to OpenI."""
    try:
        from c2net.context import upload_output
        upload_output()
        print("Results uploaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to upload results: {e}")


# ============================================================================
# DATA CONVERSION
# ============================================================================

SYSTEM_PROMPT = "You are a software engineer. Given a GitHub issue description, produce a minimal patch (unified diff format) that fixes the issue. Output ONLY the diff, nothing else."

MAX_TOTAL_CHARS = 32000  # ~8K tokens at ~4 chars/token


def convert_swelego_parquet(parquet_path: Path, output_path: Path, max_samples: int = None):
    """Convert SWE-Lego parquet to Alpaca format JSON for MindFormers.

    Output format:
    [
      {"instruction": "...", "input": "", "output": "..."},
      ...
    ]
    """
    import pandas as pd

    print(f"\nConverting SWE-Lego parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} rows")

    all_samples = []
    skipped_empty = 0
    skipped_length = 0

    for _, row in df.iterrows():
        try:
            problem_statement = row.get('problem_statement', '')
            patch = row.get('patch', '')
            hints_text = row.get('hints_text', '')

            # Skip if empty
            if not problem_statement or not patch:
                skipped_empty += 1
                continue

            # Build instruction (system prompt + problem + hints)
            instruction = f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n{problem_statement}"
            if hints_text and hints_text.strip():
                instruction += f"\n\n<hints>\n{hints_text}\n</hints>"

            # Output is the patch
            output = patch

            # Skip if too long
            total_chars = len(instruction) + len(output)
            if total_chars > MAX_TOTAL_CHARS:
                skipped_length += 1
                continue

            # Alpaca format
            all_samples.append({
                'instruction': instruction,
                'input': '',
                'output': output,
            })

            if max_samples and len(all_samples) >= max_samples:
                break

        except Exception as e:
            skipped_empty += 1
            continue

        # Progress
        if len(all_samples) % 500 == 0 and len(all_samples) > 0:
            print(f"  Converted {len(all_samples)} samples...")

    # Save as JSON array (not JSONL) for MindFormers HFDataLoader
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n  Converted: {len(all_samples)} samples")
    print(f"  Skipped (empty): {skipped_empty}")
    print(f"  Skipped (too long): {skipped_length}")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return output_path


# ============================================================================
# MINDFORMERS CONFIG GENERATION
# ============================================================================

def generate_mindformers_config(model_path: Path, data_path: Path, output_dir: Path) -> tuple:
    """Generate MindFormers YAML config file based on the official finetune_qwen3.yaml template.

    Hardware: 4× Ascend 910ProA (32 GB HBM each)
    Optimized for LoRA fine-tuning with memory constraints.

    IMPORTANT: use_flash_attention True to use native FlashAttentionScore (works on 910ProA + CANN 8.3).
    Pipeline parallelism (pp=4, mp=1) to avoid Tile sharding bug.

    Returns: (config_path, was_cached)
    """

    config_path = output_dir / 'finetune_qwen3_8b_lora.yaml'

    # Always regenerate config to pick up any changes (no caching)

    # Generate parallel_speed_up.json
    speed_up_path = output_dir / 'parallel_speed_up.json'
    if not speed_up_path.exists():
        with open(speed_up_path, 'w') as f:
            json.dump({"dataset_broadcast_opt_level": 3}, f)
        print(f"[NEW] Speed-up JSON: {speed_up_path}")
    else:
        print(f"[CACHE] Speed-up JSON: {speed_up_path}")

    yaml_content = f"""seed: 42
output_dir: '{output_dir / 'lora_output'}'
load_checkpoint: ''
load_ckpt_format: 'safetensors'
src_strategy_path_or_dir: ''
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
use_legacy: False
pretrained_model_dir: '{model_path}'

trainer:
  type: CausalLanguageModelingTrainer
  model_name: 'Qwen3'

runner_config:
  epochs: 1
  batch_size: 1
  gradient_accumulation_steps: 1

optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.0

lr_schedule:
  type: ConstantWarmUpLR
  learning_rate: 1.e-6
  warmup_ratio: 0
  total_steps: -1

train_dataset: &train_dataset
  input_columns: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  construct_args_key: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]

  data_loader:
    type: HFDataLoader
    load_func: 'load_dataset'
    path: 'json'
    data_files: '{data_path}'
    split: 'train'
    create_attention_mask: True
    create_compressed_eod_mask: False
    compressed_eod_mask_length: 128
    use_broadcast_data: True
    shuffle: True

    handler:
      - type: AlpacaInstructDataHandler
        seq_length: 4096
        padding: False
        tokenizer:
          trust_remote_code: True
          padding_side: 'right'
      - type: PackingHandler
        seq_length: 4096
        pack_strategy: 'pack'

  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  numa_enable: False
  prefetch_size: 1
  seed: 1234

train_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *train_dataset

context:
  mode: 0
  device_target: "Ascend"
  max_device_memory: "29GB"
  memory_optimize_level: "O0"
  jit_config:
    jit_level: "O0"
  ascend_config:
    precision_mode: "must_keep_origin_dtype"
    parallel_speed_up_json_path: "{speed_up_path}"

parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 4
  micro_batch_num: 4
  use_seq_parallel: False
  gradient_aggregation_group: 1

micro_batch_interleave_num: 1

parallel:
  parallel_mode: 1
  enable_alltoall: False
  full_batch: False
  dataset_strategy:
    - [1, 1]
    - [1, 1]
    - [1, 1]
    - [1, 1]
    - [1, 1, 1, 1]
  search_mode: "sharding_propagation"
  strategy_ckpt_config:
    save_file: "{output_dir / 'ckpt_strategy.ckpt'}"
    only_trainable_params: False
  enable_parallel_optimizer: False

recompute_config:
  recompute: True
  select_recompute: False
  parallel_optimizer_comm_recompute: False
  mp_comm_recompute: False

model:
  model_config:
    use_flash_attention: True
    qkv_concat: True
    hidden_dropout: 0.0
    input_sliced_sig: True
    untie_embeddings_and_output_weights: True
    position_embedding_type: "rope"
    use_contiguous_weight_layout_attention: False
    offset: 0
    params_dtype: "float32"
    compute_dtype: "float16"
    layernorm_compute_dtype: "float32"
    softmax_compute_dtype: "float32"
    rotary_dtype: "float32"
    fp32_residual_connection: True
    pet_config:
      pet_type: lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.0
      lora_a_init: 'normal'
      lora_b_init: 'zeros'
      target_modules: '.*word_embeddings|.*linear_qkv|.*linear_proj|.*linear_fc1|.*linear_fc2'
      freeze_include: ['*']
      freeze_exclude: ['*lora*']

callbacks:
  - type: MFLossMonitor
  - type: CheckpointMonitor
    prefix: "nanocoder-qwen3-8b"
    save_checkpoint_steps: 1000
    keep_checkpoint_max: 2
    integrated_save: False
    async_save: False
    checkpoint_format: "safetensors"

runner_wrapper:
  type: MFTrainOneStepCell
  scale_sense: 1.0
  use_clip_grad: True

profile: False
profile_start_step: 1
profile_stop_step: 10
init_start_profile: False
profile_communication: False
profile_memory: True
layer_scale: False
layer_decay: 0.65
lr_scale_factor: 256
"""

    with open(config_path, 'w') as f:
        f.write(yaml_content)

    print(f"[NEW] Config: {config_path}")
    return config_path, False


# ============================================================================
# TRAINING
# ============================================================================

def run_training(data_path: Path, model_path: Path, output_dir: Path, cache_status: dict):
    """Run MindFormers training on Ascend NPU."""

    start_time = time.time()
    print(f"\n{'='*60}")
    print("  Nano Coder Training (MindFormers)")
    print(f"{'='*60}")
    print(f"  Data: {data_path}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'lora_output').mkdir(parents=True, exist_ok=True)

    # Generate config (with caching)
    config_path, config_cached = generate_mindformers_config(model_path, data_path, output_dir)
    cache_status['config'] = 'cached' if config_cached else 'created'

    # Find msrun
    msrun_path = find_executable('msrun')

    # Create launcher script (always refresh - small and may change during development)
    launcher = output_dir / 'launch_train.py'
    launcher.write_text(f'''#!/usr/bin/env python3
import sys
import os
import argparse

# Enable native FlashAttention (works on 910ProA + CANN 8.3)
os.environ['MS_ENABLE_FLASH_ATTENTION'] = '1'
# os.environ['MS_ENABLE_FA_FLATTEN'] = 'off'
# os.environ['MS_DEV_GRAPH_KERNEL_FLAGS'] = '--disable_pass=FlashAttentionFusionV1,FlashAttentionFusionV2'

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--auto_trans_ckpt", default="True")
parser.add_argument("--use_parallel", default="True")
parser.add_argument("--run_mode", default="finetune")
args = parser.parse_args()

from mindformers.tools.register import MindFormerConfig
from mindformers.core.context import build_context

config = MindFormerConfig(args.config)
if args.auto_trans_ckpt.lower() == "true":
    config.auto_trans_ckpt = True
if args.use_parallel.lower() == "true":
    config.use_parallel = True
config.run_mode = args.run_mode

# Native FlashAttention handles GQA natively - no patch needed for attention.py
# Our patch handles GQA internally, replacing FlashAttentionScore kernel
print("[LAUNCHER] Using patched FlashAttention with BMM-Softmax-BMM (GQA handled internally)")

build_context(config)

from mindformers.trainer import Trainer
trainer = Trainer(args=config)
trainer.train()
''')
    print(f"[REFRESH] Launcher: {launcher}")

    # Print cache summary
    all_cached = all(v == 'cached' for v in cache_status.values())
    print(f"\n{'='*60}")
    print("  CACHE SUMMARY")
    print(f"{'='*60}")
    for name, status in cache_status.items():
        symbol = "✓" if status == 'cached' else "+"
        print(f"  {symbol} {name}: {status}")
    print(f"  * launcher: always refresh")
    if all_cached:
        print(f"\n  >>> ALL FILES FROM CACHE - FAST START <<<")
    print(f"{'='*60}\n")

    print(f"msrun: {msrun_path}")
    print("\nLaunching 4-NPU distributed training...")
    print()

    # Patch FlashAttention to use standard BMM-Softmax-BMM (910ProA has no FA kernel)
    flash_attn_path = Path('/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/mindformers/parallel_core/training_graph/transformer/flash_attention.py')
    if flash_attn_path.exists():
        backup_path = flash_attn_path.with_suffix('.py.bak')
        if not backup_path.exists():
            shutil.copy2(flash_attn_path, backup_path)
            print(f"[PATCH] Backed up: {backup_path}")

        patch_content = '''\
# Copyright 2025 Huawei Technologies Co., Ltd
# PATCHED: Use NATIVE FlashAttentionScore kernel (works on 910ProA + CANN 8.3)
"""Flash Attention Layer - Using native CANN FlashAttentionScore"""
__all__ = ['FlashAttention']

import math

import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore import ops, ParallelMode, Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.training_graph.transformer.enums import AttnMaskType
from mindformers.parallel_core.training_graph.device_matrix import layout


class FlashAttention(Cell):
    """
    FlashAttention Layer - Using NATIVE FlashAttentionScore CANN kernel.
    Supports GQA natively (Q has N1 heads, K/V have N2 heads, N1 % N2 == 0).

    Input:  Q(S,B,N1,D), K(S,B,N2,D), V(S,B,N2,D)  (SBND from attention.py)
    Output: (S,B,H) where H = N1 * D
    """

    def __init__(self,
                 config: MLATransformerConfig,
                 layer_number,
                 attn_mask_type: AttnMaskType = None,
                 attention_type: str = None,
                 attention_dropout: float = None,
                 softmax_scale: float = None,
                 cp_comm_type: str = None,
                 ):
        super(FlashAttention, self).__init__()

        if attn_mask_type:
            raise NotImplementedError("attn_mask_type not supported")
        if attention_type:
            raise NotImplementedError("attention_type unused")
        if cp_comm_type:
            raise NotImplementedError("cp_comm_type not supported")

        self.config = config
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads
        if config.multi_latent_attention:
            head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
        else:
            head_dim = projection_size // config.num_attention_heads

        self.head_num = config.num_attention_heads  # N1 = 32
        self.head_dim = head_dim                     # D = 128
        self.hidden_size = self.head_num * self.head_dim
        self.seq_length = config.seq_length          # S = 4096

        # GQA: KV heads from num_query_groups
        self.kv_num_heads = config.num_query_groups if (hasattr(config, 'num_query_groups') and config.num_query_groups and config.num_query_groups > 0) else self.head_num

        self.input_layout = "BNSD"  # Native FA supports BNSD with GQA
        self.sparse_mode = config.sparse_mode
        self.attention_dropout = config.attention_dropout if attention_dropout is None else attention_dropout
        scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
        self.keep_prob = 1.0 - self.attention_dropout
        self.scalar_value = scale

        self.use_alibi_mask = config.use_alibi_mask
        self.use_ring_attention = config.use_ring_attention

        # Ops
        self.transpose_op = ops.Transpose()
        self.reshape_op = ops.Reshape()
        self.cast_op = ops.Cast()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard(config)

        print(f"[NATIVE FA v8] layer={layer_number} q_heads={self.head_num} "
              f"kv_heads={self.kv_num_heads} dim={self.head_dim} seq={self.seq_length} "
              f"layout={self.input_layout} scale={self.scalar_value:.6f}")

    def construct(self, query, key, value, attention_mask,
                  attn_mask_type=None, attention_bias=None, packed_seq_params=None,
                  alibi_mask=None, prefix=None, padding_mask=None,
                  actual_seq_qlen=None, actual_seq_kvlen=None):
        # Input from attention.py: (S, B, N, D) in SBND layout
        # FlashAttentionScore with BNSD expects: (B, N, S, D)
        q = self.transpose_op(query, (1, 2, 0, 3))  # (B, N1, S, D)
        k = self.transpose_op(key,   (1, 2, 0, 3))  # (B, N2, S, D)
        v = self.transpose_op(value, (1, 2, 0, 3))  # (B, N2, S, D)

        # Ensure float16 (FlashAttentionScore requires fp16 or bf16)
        q = self.cast_op(q, mstype.float16)
        k = self.cast_op(k, mstype.float16)
        v = self.cast_op(v, mstype.float16)

        # Convert attention_mask: MindFormers uses additive mask (0=keep, large_neg=mask)
        # FlashAttentionScore uses bool/uint8 mask (0/False=keep, 1/True=mask)
        attn_mask = None
        if attention_mask is not None:
            # attention_mask has large negative values where tokens should be masked
            # Convert to bool: True where masked (value < -1)
            attn_mask = self.cast_op(attention_mask < -1.0, mstype.uint8)

        # Call native FlashAttentionScore kernel
        # GQA is handled natively: N1=32 query heads, N2=8 KV heads
        out = ops.flash_attention_score(
            q, k, v,
            head_num=self.head_num,
            attn_mask=attn_mask,
            keep_prob=self.keep_prob,
            scalar_value=self.scalar_value,
            input_layout=self.input_layout,
            sparse_mode=0,
        )

        # Output: (B, N1, S, D) in BNSD
        # Convert back to (S, B, H) for attention.py
        out = self.transpose_op(out, (2, 0, 1, 3))  # (S, B, N1, D)
        out = self.reshape_op(out, (self.seq_length, -1, self.hidden_size))
        return out

    def shard(self, config):
        pass  # FlashAttentionScore handles its own sharding

    def sharding_propagation(self, config):
        pass
'''

        flash_attn_path.write_text(patch_content)
        print(f"[PATCH] Replaced FlashAttention with NATIVE FlashAttentionScore (v8): {flash_attn_path}")

        # Clear pycache to ensure new code is loaded
        pycache_dir = flash_attn_path.parent / '__pycache__'
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir, ignore_errors=True)
            print(f"[PATCH] Cleared pycache: {pycache_dir}")
    else:
        print(f"[WARN] FlashAttention file not found at {flash_attn_path}")

    # Clear ALL graph caches to ensure recompilation with patched code
    print("[CLEAN] Clearing graph caches...")
    import glob as _glob
    for cache_pattern in ['/home/ma-user/work/rank_*/graph_cache',
                          '/home/ma-user/work/output/rank_*/graph_cache',
                          '/home/ma-user/graph_cache']:
        for cache_dir in _glob.glob(cache_pattern):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[CLEAN] Removed {cache_dir}")
    # Disable cache during development to always recompile
    os.environ['MS_COMPILER_CACHE_ENABLE'] = '0'
    print("[CLEAN] Disabled MS_COMPILER_CACHE_ENABLE for fresh compilation")

    try:
        cmd = [
            msrun_path,
            '--bind_core=True',
            '--worker_num=4',
            '--local_worker_num=4',
            '--master_port=8118',
            '--log_dir=' + str(output_dir / 'msrun_log'),
            '--join=True',
            '--cluster_time_out=7200',
            str(launcher),
            '--config', str(config_path),
            '--auto_trans_ckpt', 'True',
            '--use_parallel', 'True',
            '--run_mode', 'finetune',
        ]
        subprocess.run(cmd, check=True)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{'='*60}")
        print("  Training Complete!")
        print(f"  Output: {output_dir / 'lora_output'}")
        print(f"  Duration: {duration/60:.1f} minutes")
        print(f"  Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"{'='*60}")

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n{'='*60}")
        print(f"  Training FAILED after {duration/60:.1f} minutes")
        print(f"  Exit code: {e.returncode}")
        # Print worker logs for debugging
        log_dir = output_dir / 'msrun_log'
        for log_file in sorted(log_dir.glob('worker_*.log')):
            print(f"\n--- {log_file.name} (ERROR/WARNING lines + last 50) ---")
            try:
                lines = log_file.read_text().splitlines()
                # Print all ERROR lines first
                error_lines = [l for l in lines if 'ERROR' in l or 'RuntimeError' in l or 'Traceback' in l or 'raise' in l]
                if error_lines:
                    print("=== ERRORS FOUND ===")
                    for line in error_lines[:50]:
                        print(line)
                # Then last 50 lines
                print("=== Last 50 LINES ===")
                for line in lines[-50:]:
                    print(line)
            except Exception as read_err:
                print(f"  Could not read log: {read_err}")
        print(f"{'='*60}")
        raise
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        raise


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Nano Coder Training on OpenI (MindFormers)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to use (for testing)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  Nano Coder - OpenI Training (MindFormers)")
    print(f"{'='*60}\n")

    context = None
    cache_status = {}  # Track what was cached vs created

    try:
        # Initialize c2net context (required on OpenI)
        context = init_c2net()

        # Get paths from c2net context
        model_path = get_model_path(context)
        print(f"Model: {model_path}")

        data_path = get_dataset_path(context)
        print(f"Dataset: {data_path}")

        # Get output directory from c2net context
        if not hasattr(context, 'output_path'):
            raise RuntimeError("Output path not found in c2net context.")

        output_dir = Path(context.output_path)
        print(f"Output: {output_dir}")

        # Convert parquet to JSON if needed (with caching)
        if data_path.suffix == '.parquet':
            # Use consistent filename for caching
            json_path = output_dir / 'swelego_alpaca.json'
            if json_path.exists():
                print(f"[CACHE] Dataset: {json_path}")
                cache_status['dataset'] = 'cached'
                data_path = json_path
            else:
                json_path = convert_swelego_parquet(data_path, json_path, args.max_samples)
                cache_status['dataset'] = 'created'
                data_path = json_path
        elif data_path.suffix == '.jsonl':
            # Convert JSONL to JSON array for MindFormers (with caching)
            json_path = output_dir / 'swelego_alpaca.json'
            if json_path.exists():
                print(f"[CACHE] Dataset: {json_path}")
                cache_status['dataset'] = 'cached'
                data_path = json_path
            else:
                print(f"\nConverting JSONL to JSON array: {data_path}")
                samples = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            # Convert from ShareGPT to Alpaca format
                            convs = sample.get('conversations', [])
                            if len(convs) >= 2:
                                samples.append({
                                    'instruction': convs[0].get('value', ''),
                                    'input': '',
                                    'output': convs[1].get('value', ''),
                                })
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
                print(f"[NEW] Dataset: {json_path} ({len(samples)} samples)")
                cache_status['dataset'] = 'created'
                data_path = json_path

        # Run training
        run_training(data_path, model_path, output_dir, cache_status)

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"  ERROR: {type(e).__name__}")
        print(f"  {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Always try to upload results
        if context is not None:
            print("\nUploading results to OpenI...")
            upload_results(context)


if __name__ == '__main__':
    main()
