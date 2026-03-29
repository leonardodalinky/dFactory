# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This project is a research project based on the offical LLaDA2 MoE model implementation. It should create a custom training pipeline using LLaDA2 to support Latent-Block Diffusion Language model.

For the idea of Latent-Block Diffusion Language model, please refer to @Idea_for_Latent-Block_dLLM.md in the project root directory.

## Project Overview

- `configs/`: YAML training configs and JSON model configs
- `models/llada2_moe/`: LLaDA2 MoE model architecture
- `scripts/`: model conversion (merged ↔ MoE format), HF download utilities
- `tasks/`: training entry points and dataset pipeline
  - `train_llada2_bd.py`: full fine-tuning with block diffusion
  - `dataset/`: data loading and preprocessing transforms
- `tokenizers/latent/`: custom tokenizer for latent block diffusion
- `VeOmni/`: git submodule — distributed training framework (FSDP2, parallelism, checkpointing)
- `train.sh`: torchrun launcher (auto-detects GPUs)

## Running training

Always set `PYTHONPATH` to include the VeOmni submodule:

```shell
# Full fine-tuning (interactive or via train.sh)
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH \
  bash train.sh tasks/train_llada2_bd.py configs/sft/icoder_mini_bd_sft.yaml
```

## Environment

The project should be synced to William & Mary HPC cluster. You should be aware of the Claude skills related to W&M HPC cluster, such as `wm-remote-dev`.
