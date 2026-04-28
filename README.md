# brain-tumor-agent

Active diagnostic reasoning agent for 3D brain tumor MRI — Phase 1 implementation.

## Overview

This repository implements an LLM-based agent that performs **active acquisition** of diagnostic evidence for brain tumor cases. Rather than ingesting all available evidence at once (the static MLLM paradigm), the agent decides at each step which specialist tool to invoke (segmentation, feature extraction, molecular profiling, etc.), updates its belief over differential diagnoses, and commits to a final diagnosis when sufficient evidence has accumulated.

The architecture follows a POMDP formulation with Qwen3-14B as the policy LLM, validated in Phase 0 to match Claude Opus on this domain (96% vs 94% on BraTS 2021).

## Phase 0 baselines (already established)

| System | True diagnosis accuracy | Notes |
|---|---|---|
| M3D-LaMed (FLAIR / T1ce) | 0% | end-to-end MLLM, hallucinates |
| MedGemma (FLAIR / T1ce) | 4% / 24% | end-to-end medical MLLM |
| MedGemma + KAIST features | 26% | permutation-validated |
| Qwen3-14B + KAIST features | **96%** | local, 4-bit |
| Claude Opus + KAIST features | 92-94% | reference |

KAIST nnU-Net Dice on BraTS 2021 (n=500): WT 0.956 / TC 0.931 / ET 0.811.

See `docs/phase0_findings.md` for the full report.

## Quick start

```bash
# 1. Enter
cd brain-tumor-agent

# 2. Create env (uv recommended)
uv venv --python 3.11
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate

# 3. Install
uv pip install -e ".[dev]"

# 4. Verify
python -c "import src; print('OK', src.__version__)"
pytest tests/ -q

# 5. Configure data paths
cp .env.example .env
# edit .env to point at BraTS / UPenn-GBM / KAIST predictions
```

## Repository layout

```
brain-tumor-agent/
├── src/
│   ├── tools/         # Specialist tool wrappers (Step 1.2)
│   ├── simulators/    # Simulated observation providers (Phase 2)
│   ├── belief/        # Belief state updates (Step 1.5)
│   ├── policy/        # Reasoning policy = LLM (Step 1.5)
│   ├── grounding/     # Evidence grounding (Phase 2)
│   ├── eig/           # Expected information gain (Phase 2)
│   ├── agent/         # Main agent loop (Step 1.5)
│   ├── eval/          # Evaluation harness (Step 1.8)
│   └── train/         # SFT + RL training (Step 1.6, 1.7)
├── configs/           # YAML configs for experiments
├── data/              # symlinks ONLY, no real data committed
├── experiments/       # outputs: logs, checkpoints, results
├── notebooks/         # exploratory work
├── tests/             # unit + integration tests
├── scripts/           # data prep, batch runs, helpers
└── docs/              # phase0_findings, design notes
```

## Phase 1 milestone schedule

| Month | Deliverable |
|---|---|
| 1 | Repo + 5 core tools + vLLM/Qwen3 stack |
| 2 | Agent reasoning loop + integration tests |
| 3 | ACTMED-Imaging trajectory dataset + SFT cold-start |
| 4 | RL with process rewards + UPenn-GBM Pareto evaluation |

See `docs/playbook.md` for step-by-step DoD criteria.

## Citation / acknowledgments

KAIST nnU-Net (Luu & Park, BrainLes 2021). Qwen3 (Alibaba, 2025). BraTS 2021. UPenn-GBM (TCIA, CC-BY).
