#!/bin/bash
# Launches a short (3000-step) fresh-anneal run, warm-started from
# v3b/step_24000 (our best confirmed checkpoint), with beta_language pushed
# further down (0.2 -> 0.1) to give Kalenjin/Maasai/Somali more relative
# training exposure. See config_1b_v4.yaml for full rationale.
set -e
source /etc/rp_environment
source ~/venv/bin/activate
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
# WANDB_API_KEY already exported via /etc/rp_environment

OUTPUT_DIR=/root/runs/ctc1b_v4
cd ~/omnilingual-asr
python -u -m workflows.recipes.wav2vec2.asr "$OUTPUT_DIR" --config-file /root/training/config_1b_v4.yaml
