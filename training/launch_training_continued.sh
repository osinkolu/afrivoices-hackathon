#!/bin/bash
# Launches the short fresh-anneal continuation run: warm-starts from step_8000
# (trained on mojibake-corrupted text for kik/luo/som) with the text-correction
# patch now applied in the data pipeline, so the model adapts to correctly-
# encoded targets. See config_1b_continued.yaml for why num_steps is a fresh,
# realistic budget rather than resuming the original run's step count.
set -e
source /etc/rp_environment
source ~/venv/bin/activate
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
# WANDB_API_KEY already exported via /etc/rp_environment

OUTPUT_DIR=/root/runs/ctc1b_continued
cd ~/omnilingual-asr
python -u -m workflows.recipes.wav2vec2.asr "$OUTPUT_DIR" --config-file /root/training/config_1b_continued.yaml
