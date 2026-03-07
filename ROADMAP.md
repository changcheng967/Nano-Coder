# Nano Coder — Roadmap to the World's Best Small Coding Model

> **Goal:** Best-in-class 8B coding agent, scaling to 32B to approach frontier
> **Hardware:** 4×Ascend 910ProA NPU (32GB each, 128GB total), OpenI platform
> **Benchmark:** SWE-bench Verified (500 real GitHub issues)
> **Last Updated:** 2026-03-07

## The Battlefield: SWE-bench Verified (March 2026)

### Frontier Models (Custom Scaffolds)
| Rank | Model | Score | Parameters | Cost |
|------|-------|-------|------------|------|
| 1 | Claude Sonnet 5 | 82.1% | ~200B+ | $$$$ |
| 2 | Claude Opus 4.5 | 80.9% | ~200B+ | $$$$ |
| 3 | Claude Opus 4.6 | 80.8% | ~200B+ | $$$$ |
| 4 | Gemini 3.1 Pro | 80.6% | ~300B+ | $$$$ |
| 5 | MiniMax M2.5 | 80.2% | unknown | $$$$ |
| 6 | GPT-5.2 | 80.0% | ~200B+ | $$$$ |

### Standardized Leaderboard (mini-SWE-agent, same scaffold for all)
| Rank | Model | Score | Parameters |
|------|-------|-------|------------|
| 1 | Claude Opus 4.5 | 76.8% | ~200B+ |
| 2 | Gemini 3 Flash | 75.8% | unknown |
| 3 | Claude Opus 4.6 | ~76% | ~200B+ |
| 4 | Gemini 3 Pro | 74% | ~300B+ |

### Open-Weight Models
| Model | Score | Type | Parameters |
|-------|-------|------|------------|
| CoderForge-32B | 59.4% | SFT only, pass@1 | 32B |
| DeepSWE-Preview | 59.0% | RL only + TTS@16 | 32B |
| DeepSWE-Preview | 42.2% | RL only, pass@1 | 32B |
| Devstral-Small | 46.6% | Agent | 24B |
| CoderForge-4B | 43.0% | SFT only, pass@1 | 4B |
| SWE-Agent-LM | 40.2% | Agent | 32B |
| Klear-AgentForge | ~35% | SFT + RL | **8B** |
| CodePilot (MCTS) | 24.7% | MCTS + Qwen3-8B | **8B** |
| SkyRL-Agent | 21.6% | Agent | 14B |

### The Gaps That Define Our Opportunity
```
CoderForge-4B (SFT only) = 43.0%   ← 4B model, just SFT, no tricks
Klear-AgentForge-8B      = ~35%    ← current 8B SOTA
CodePilot-8B (MCTS)      = 24.7%   ← 8B with search

Nobody has combined: 8B + CoderForge data + RL + TTS + MCTS + verify-on-edit
That combination is Nano Coder.
```

## Our Record to Claim

**"Best 8B coding agent system in the world"**

Current 8B SOTA: Klear-AgentForge ~35% (SFT+RL, no TTS)
Our target: **65-70%** (full compound system)
That would beat every open-weight model up to 32B.

If we scale to 32B with QLoRA: target **75-80%** (frontier territory).

## What Changed: The CoderForge Bombshell (Feb 25, 2026)

Eight days ago, Together AI released CoderForge-Preview:
- **258K test-verified trajectories** (155K passing, 103K failing)
- 51K tasks across 1,655 repositories
- OpenHands format, 128K context, Qwen3-Coder-480B as teacher
- **72.8 GB** on HuggingFace, parquet format, permissively licensed
- SFT-only results: Qwen3-4B → **43.0%**, Qwen3-32B → **59.4%**
- Cost to generate: $130K. Cost for us to use: **$0** (it's open)

This changes everything. The SWE-Lego dataset (5K trajectories, 330MB)
is a toy compared to this. CoderForge is the training data for NC.

Also 18 hours ago: adding "verify after every edit" to a 3B-active model
went from 22% → 38% on SWE-bench Verified Hard. A scaffold trick.
Zero training. This validates our compound system approach.

## The Nano Coder Compound System

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NANO CODER SYSTEM                               │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Layer 4: NC-Search (MCTS / Best-of-N)                       │  │
│  │  Generate 8-16 candidates, tree-search over edits             │  │
│  │  Proven: +15-20% (S*, CodePilot, DeepSWE TTS)               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ▲                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Layer 3: NC-Verify (Execution-Based + PRM Verifier)         │  │
│  │  Run tests on candidates, PRM scores reasoning quality        │  │
│  │  Proven: +10-17% (DeepSWE hybrid, GenPRM)                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ▲                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Layer 2: NC-Scaffold (Verify-on-Edit + Phased Protocol)     │  │
│  │  8-phase problem solving, test after every edit, repo index   │  │
│  │  Proven: +10-16% (Qwen3.5 verify-on-edit, CoderForge prompt)│  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ▲                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Layer 1: NC-Policy (Fine-Tuned Qwen3-8B / 32B)             │  │
│  │  SFT on CoderForge → RL with GRPO++ on R2E-Gym              │  │
│  │  Proven: 43% (4B SFT), 59% (32B SFT), 42% (32B RL)         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Each layer multiplies performance. They are built independently and
composed at inference time. Training happens only in Layer 1 and
partially Layer 3.

## Revised Roadmap

### NC-1 Preview (NOW — 2h debug task) — Pipeline Validation

**Status:** Infrastructure ready, testing on OpenI

**What it does:** SFT on SWE-Lego issue→patch pairs (single-turn).
This proves the OpenI pipeline works: c2net data loading, MindFormers
training, model checkpointing.

**Infrastructure completed:**
- [x] C2NET context integration (model/data/output paths)
- [x] MindFormers YAML config generation
- [x] 4-NPU pipeline parallelism (mp=1, pp=4)
- [x] FlashAttention disabled (910ProA unsupported)
- [x] FlashAttention patched with BMM-Softmax-BMM
- [x] GQA support with static reshape dims

**Ascend 910ProA Compatibility Fixes:**
```
Problem: FlashAttentionScore kernel unavailable on 910ProA (only 910B)

Solutions applied:
1. MS_ENABLE_FLASH_ATTENTION=0 (env var)
2. MS_DEV_GRAPH_KERNEL_FLAGS=--disable_pass=FlashAttentionFusionV1,V2
3. use_flash_attention: False in config.json
4. Monkey-patch Qwen3Config and TransformerConfig
5. Patch flash_attention.py with BMM-Softmax-BMM implementation
```

**Expected score:** ~5-10% (single-turn format can't solve real SWE tasks)

**Value:** Infrastructure validation. Every future stage depends on this
working end-to-end.

---

### NC-1 (Week 1) — Foundation: 8B SFT on CoderForge — Target: 40-48%

**Goal:** Train the best 8B coding model that has ever existed via SFT.

**Data:** CoderForge-Preview (155K passing trajectories, 72.8GB)
- Download from HuggingFace, upload to OpenI
- Already in OpenHands multi-turn format
- Already tokenized (trajectories-tokenized subset available)
- Median trajectory: 38K tokens, average 41K tokens

**Model:** Qwen3-8B + LoRA (rank 8, attention layers)

**Training config (MindFormers):**
```yaml
# Parallelism
model_parallel: 1          # mp=1 avoids Tile sharding bug
pipeline_stage: 4          # pp=4 uses all 4 NPUs
micro_batch_num: 4

# Model
pet_type: lora
lora_rank: 8
lora_alpha: 16
target_modules: '.*word_embeddings|.*linear_qkv|.*linear_proj|.*linear_fc1|.*linear_fc2'

# Sequence
seq_length: 4096
compute_dtype: float16

# FlashAttention (910ProA unsupported)
use_flash_attention: False
```

**Memory estimate:** Qwen3-8B in bf16 = ~16GB, LoRA adapters = ~2GB,
optimizer states = ~4GB, activations with gradient checkpointing = ~8GB.
Total: ~30GB. Fits on single Ascend 910 (32GB).
With 4 NPUs: data parallel, 4× throughput.

**Timeline:**
- Day 1: Download CoderForge, upload to OpenI, adapt train.py
- Day 2-5: Train 3 epochs on 155K trajectories at 32K context
- Day 5-6: Evaluate on SWE-bench Verified (subset first, then full)

**Expected result:** 40-48% pass@1
- CoderForge-4B = 43.0% with 5 epochs
- 8B should exceed 4B by 3-8%
- If we hit 48%, that's a new 8B world record

**Why this beats current 8B SOTA:**
- Klear-AgentForge-8B used only 66K SWE-Smith trajectories
- We use 155K CoderForge trajectories (higher quality, more diverse)
- Same base model (Qwen3-8B), much better data

---

### NC-2 (Week 2-3) — Scaffold Engineering — Target: 55-62%

**Goal:** Add NC-Scaffold and NC-Verify for +12-18% over raw model.
This is pure engineering, zero additional training.

**NC-Scaffold components:**

**2a. Verify-on-Edit Loop** (proven: +16% on Qwen3.5-35B-A3B)
```python
# After every code edit action:
if action_type == "edit":
    test_result = run_relevant_tests(repo_path)
    if test_result.failed:
        observation += f"\n\nTests failed after edit:\n{test_result.stderr}"
        observation += f"\nFailing test: {test_result.failing_test}"
        observation += "\nPlease fix the issue before proceeding."
```

**2b. 8-Phase Problem-Solving Protocol** (from CoderForge prompt template)
```
Phase 1: READING — reword the issue, highlight errors, stack traces
Phase 2: RUNNING — install and run existing tests
Phase 3: EXPLORATION — grep for relevant methods, classes, files
Phase 4: TEST CREATION — write reproduction script before fixing
Phase 5: FIX ANALYSIS — state problem, location, and fix strategy
Phase 6: FIX IMPLEMENTATION — minimal focused changes
Phase 7: VERIFICATION — run reproduction + edge cases + existing tests
Phase 8: FINAL REVIEW — re-read issue, verify all requirements met
```

**2c. Codebase Pre-Indexing**
```python
# Before agent starts, provide a condensed repo map:
# - All .py files with class/function signatures
# - README.md summary
# - Test file locations
# - Import graph (what imports what)
repo_map = build_repo_index(repo_path)
system_prompt += f"\n\nRepository structure:\n{repo_map}"
```

**2d. Execution-Based Verification (NC-Verify, no training needed)**
```python
# Generate N=8 candidate patches with temperature=0.7
# Run full test suite on each
# Select candidate passing the most tests
candidates = [generate_patch(problem, temp=0.7) for _ in range(8)]
scores = [run_tests(repo, candidate) for candidate in candidates]
best = candidates[argmax(scores)]
```

**Timeline:**
- Day 7-8: Fork mini-swe-agent, implement verify-on-edit
- Day 9-10: Add 8-phase protocol + codebase indexing
- Day 11-12: Implement parallel candidate generation + test selection
- Day 13-14: Evaluate full NC-2 system on SWE-bench Verified

**Expected result:**
- NC-1 pass@1: ~45%
- + verify-on-edit + phased protocol: ~52-55%
- + pass@8 execution verification: ~58-62%

**At 60%+, NC would beat DeepSWE-Preview (59%) — a 32B model with RL.**
We'd achieve this with an 8B model and SFT only + scaffold + TTS.

---

### NC-3 (Week 3-5) — RL Fine-Tuning with GRPO++ — Target: 52-58% pass@1

**Goal:** Apply reinforcement learning on top of NC-1 checkpoint to
improve the base policy model. This is independent of scaffold gains.

**Why RL after SFT:** DeepSWE showed pure RL on Qwen3-32B went from
23% → 42% in 200 steps. Starting from our SFT checkpoint (~45%),
RL should push to 52-58% pass@1 (before scaffold/TTS).

**RL Recipe (GRPO++ from DeepSWE):**
- Clip high (from DAPO): upper PPO clip = 1.28 (encourages exploration)
- No KL loss: don't constrain to SFT distribution
- No reward std dev (Dr.GRPO): removes difficulty bias
- Length normalization: divide loss by max context length
- Leave-one-out (RLOO): reduce variance without bias
- Compact filtering: mask trajectories hitting max context/steps/timeout
- No entropy loss: prevents training collapse

**Reward:** Binary 0/1. Run test suite on model's patch.
Pass all tests = 1, fail any = 0.

**Environment:** SWE-Smith Docker containers (37K tasks available).
Each rollout spawns a Docker container, model interacts via bash,
reward is test execution result.

**Challenge:** Docker orchestration on OpenI. If not feasible,
use offline RL: generate rollouts locally, label with rewards,
train on the labeled data.

**Offline RL alternative:**
1. Use NC-1 to generate 8 rollouts per problem on 5K R2E-Gym tasks
2. Run tests to label each rollout as pass/fail
3. Use DPO (Direct Preference Optimization) with pass>fail pairs
4. This avoids needing live Docker environments on OpenI

**Timeline:**
- Week 3: Set up RL pipeline (online GRPO++ or offline DPO)
- Week 4: Train 200-500 RL iterations
- Week 5: Evaluate and iterate

**Expected pass@1 improvement:** +7-13% over SFT baseline

---

### NC-4 (Week 5-8) — Self-Improvement Loop — Target: 58-65% pass@1

**Goal:** Use NC-3 to generate its own training data, filter for
quality, and retrain. Iterative self-improvement.

**Pipeline:**
```
Iteration 1:
  1. NC-3 generates 8 rollouts on 37K SWE-Smith tasks
  2. Run tests → ~55% solve rate → ~20K new passing trajectories
  3. Filter: keep only passing, <64K tokens, <50 steps
  4. Retrain NC-3 on original CoderForge + new self-generated data
  5. → NC-4-iter1

Iteration 2:
  1. NC-4-iter1 generates on remaining unsolved tasks + new repos
  2. Higher solve rate → more diverse training data
  3. Retrain → NC-4-iter2

Repeat 3-5 iterations.
```

**Why this works:**
- SWE-RL (Meta) showed self-play adds +10.4 points
- CoderForge showed more data = better (155K >> 66K >> 5K)
- Each iteration adds trajectories from problems the model
  previously couldn't solve, expanding capability

**Compute:** Inference for rollout generation is cheap on 4×Ascend 910.
Retraining is another SFT run (same as NC-1, slightly larger dataset).

**Timeline:** 3-4 weeks (mostly waiting for rollout generation)

---

### NC-5 (Week 8-12) — Scale to 32B — Target: 65-75% pass@1

**Goal:** Apply everything learned from 8B to Qwen3-32B.

**Feasibility on 4×Ascend 910 (128GB total):**
- Qwen3-32B in bf16 = ~64GB (fits across 4 NPUs with FSDP)
- QLoRA (4-bit base + 16-bit LoRA): ~20GB per NPU
- With gradient checkpointing: fits with batch_size=1

**Training data:** CoderForge-Preview (same 155K trajectories)
+ NC-4's self-generated trajectories (~20-50K additional)

**Expected improvement over 8B:**
- CoderForge showed: 4B → 43%, 32B → 59.4% (+16.4 points)
- If NC-8B = 58% pass@1, NC-32B should be ~70-75% pass@1

**With NC-Scaffold + NC-Verify (pass@8):** 75-80%

**This enters frontier territory.**

---

### NC-6 (Month 3-4) — MCTS + PRM + Advanced TTS — Target: 75-82%

**Goal:** Build the full search and verification stack.

**NC-Search (MCTS over edit trajectories):**
- At each agent step, generate K=4 candidate actions
- Use PRM to score each action's reasoning quality
- Expand the best branch, prune the worst
- Backtrack from dead ends (failed tests, wrong files)
- CodePilot showed MCTS + 8B = 24.7% from scratch
- On top of our fine-tuned model, MCTS should give +5-10%

**NC-PRM (Process Reward Model):**
- Train on CoderForge pass/fail trajectory pairs (155K pass, 103K fail)
- Label each step as correct/incorrect using test execution signals
- Fine-tune Qwen3-8B LoRA as step-level reward model
- Use for MCTS node evaluation + best-of-N selection

**NC-Ensemble:**
- Train 3-4 LoRA adapters with different hyperparameters
- Each has different failure modes
- At inference: run all 4, select best via execution verification
- Low-effort high-reward technique

**Expected:** Pass@16 with MCTS + PRM + ensemble = 75-82%

---

### NC-7 (Month 4+) — Novel Research — Target: 80%+

**These are the research bets. Some will work, some won't.**

**NC-Memory:** Persistent codebase knowledge across problems in the
same repository. After solving 5 Django issues, the model knows Django
patterns. Implemented as a retrieval-augmented LoRA merge or vector DB.
Nobody has done this for SWE-bench.

**NC-Adaptive:** Meta-controller that allocates thinking budget per step.
100 tokens for "run this test," 3000 tokens for "diagnose root cause."
DeepSWE showed this emerges from RL; we make it explicit and aggressive.

**NC-Compiler:** Model outputs structured repair specifications instead
of raw code edits. Deterministic compiler translates spec → patch.
Eliminates syntactic errors while preserving semantic intelligence.

**NC-TTT (Test-Time Training):** Before each problem, fine-tune a
temporary LoRA on the specific repository's codebase. The model
briefly specializes on Django (or matplotlib, or scikit-learn)
before attempting the fix. ~1 minute per problem overhead.

## Score Projections

### Conservative / Realistic / Optimistic

| Stage | Model | Technique | Conservative | Realistic | Optimistic |
|-------|-------|-----------|-------------|-----------|------------|
| NC-1 Preview | 8B | SFT (SWE-Lego, single-turn) | 3% | 5% | 10% |
| NC-1 | 8B | SFT (CoderForge, multi-turn) | 38% | 45% | 48% |
| NC-2 | 8B | NC-1 + Scaffold + TTS@8 | 50% | 58% | 65% |
| NC-3 | 8B | NC-2 base + RL (GRPO++) | 52% | 58% | 62% |
| NC-4 | 8B | NC-3 + Self-improvement loop | 56% | 62% | 68% |
| NC-5 | 32B | QLoRA SFT + RL + all scaffolding | 68% | 75% | 80% |
| NC-6 | 32B | NC-5 + MCTS + PRM + ensemble | 73% | 78% | 83% |
| NC-7 | 32B | NC-6 + novel research | 76% | 80% | 85%+ |

### World Records We Can Claim at Each Stage

| Stage | Record |
|-------|--------|
| NC-1 | Best 8B model via SFT on SWE-bench Verified (beat Klear ~35%) |
| NC-2 | Best 8B compound system (beat CodePilot 24.7%, beat all ≤14B) |
| NC-3 | Best 8B model pass@1 (potential new absolute 8B SOTA) |
| NC-4 | Best 8B self-improving agent |
| NC-5 | Best open-weight 32B system (beat DeepSWE 59%, CoderForge 59.4%) |
| NC-6 | Competitive with frontier models |
| NC-7 | Potentially exceeding some frontier models |

## Compute Budget

| Stage | Training Time | Inference/Eval Time | Total |
|-------|--------------|--------------------| ------|
| NC-1 Preview | 2 hours | 1 hour | 3 hours |
| NC-1 | 4-5 days | 1 day | ~1 week |
| NC-2 | 0 (scaffold only) | 2 days eval | 2 days |
| NC-3 | 1-2 weeks (RL) | 1 day eval | ~2 weeks |
| NC-4 | 2-3 weeks (rollouts + retrain) | 2 days eval | ~3 weeks |
| NC-5 | 1-2 weeks (32B training) | 2 days eval | ~2 weeks |
| NC-6 | 1 week (PRM training) | 3 days eval | ~10 days |
| NC-7 | ongoing | ongoing | ongoing |

**Total to NC-5 (frontier competitive): ~2 months**
**Total to NC-6 (frontier matching): ~3 months**

All on free OpenI compute. $0 GPU cost.

## Key Data Assets

| Dataset | Size | Source | Use |
|---------|------|--------|-----|
| CoderForge-Preview | 72.8GB, 155K pass / 103K fail | HuggingFace | NC-1 SFT training |
| SWE-Lego Real Data | 330MB, 5K trajectories | HuggingFace | NC-1 Preview (current) |
| SWE-Smith | 37K tasks, Docker envs | GitHub | NC-3 RL environments |
| R2E-Gym | 4.2K tasks, Docker envs | GitHub | NC-3 RL environments |
| Self-generated | Growing, 20-50K per iteration | Us | NC-4 self-improvement |

## Critical Dependencies

### Must Have
- [x] OpenI account with Ascend 910ProA access
- [x] Qwen3-8B base model uploaded to OpenI
- [x] SWE-Lego dataset uploaded
- [x] train.py pipeline with MindFormers (NC-1 Preview)
- [x] FlashAttention patched for 910ProA compatibility
- [x] 4-NPU pipeline parallelism configured
- [ ] CoderForge-Preview downloaded and uploaded to OpenI
- [ ] train.py updated for OpenHands multi-turn format

### Should Have
- [ ] mini-swe-agent fork with verify-on-edit
- [ ] SWE-bench Verified evaluation harness
- [ ] Docker containers for RL training (NC-3)

### Nice to Have
- [ ] Qwen3-32B uploaded to OpenI (NC-5)
- [ ] Kubernetes for parallel rollout collection (NC-3+)
- [ ] PRM training pipeline (NC-6)

## What Makes This Different from Every Other Attempt

1. **Nobody has combined all known techniques on 8B.**
   Klear did SFT+RL but not TTS. CodePilot did MCTS but not SFT.
   DeepSWE did RL+TTS but only 32B. We do everything on 8B.

2. **CoderForge-Preview is 8 days old.**
   The dataset that gets 4B to 43% just became public.
   We're among the first to use it for 8B training.

3. **Scaffold innovations compound with model improvements.**
   Verify-on-edit alone gives +16%. The phased protocol gives +5%.
   These multiply with better base models, not add.

4. **$0 compute cost.**
   Every competing result (DeepSWE: ~$100K, CoderForge: $130K data gen)
   required massive GPU budgets. We use free OpenI Ascend 910s.
   If we match their results, that's a story in itself.

## Immediate Next Steps

1. **Right now:** Let NC-1 Preview finish its 2-hour training run
2. **Today:** Download CoderForge-Preview, begin upload to OpenI
3. **Today:** Update train.py for OpenHands trajectory format
4. **Tomorrow:** Launch NC-1 real training (CoderForge, 32K context, 3 epochs)
5. **This week:** Fork mini-swe-agent, implement verify-on-edit

## Sources

- [CoderForge-Preview](https://www.together.ai/blog/coderforge-preview) — 59.4% with SFT-only on 32B (Feb 2026)
- [DeepSWE](https://www.together.ai/blog/deepswe) — 59% with RL-only + TTS on 32B (Jul 2025)
- [Klear-AgentForge](https://arxiv.org/abs/2511.05951) — 8B SOTA via SFT + RL (Nov 2025)
- [CodePilot](https://arxiv.org/pdf/2602.00129) — MCTS + Qwen3-8B = 24.7% (Jan 2026)
- [S*](https://arxiv.org/abs/2502.14382) — Test-time scaling, 3B beats GPT-4o-mini (Feb 2025)
- [SWE-Search](https://arxiv.org/abs/2410.20285) — MCTS for SWE, +23% relative (Oct 2024)
- [Qwen3.5 verify-on-edit](https://reddit.com/r/LocalLLaMA/) — 22%→38% with scaffold trick (Mar 2026)
- [SWE-bench leaderboard](https://llm-stats.com/benchmarks/swe-bench-verified) — Current scores (Mar 2026)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — 100-line baseline scaffold
- [SWE-RL](https://arxiv.org/abs/2502.18449) — RL for SWE with GRPO (Feb 2025)

---

*"The best 8B coding agent the world has ever seen —*
*trained for $0, built in public, open to everyone."*

*That's a paper. That's a contribution. That's Nano Coder.*