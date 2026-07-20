#!/bin/bash
# Launches the longer (6000-step) fresh-anneal run, warm-started from
# step_2000 (the mojibake-fix run's best checkpoint, confirmed ~21% lower
# dev-set WER than step_8000). See config_1b_continued_v2.yaml for why
# num_steps is bounded to a realistic budget.
set -e
source /etc/rp_environment
source ~/venv/bin/activate
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
# WANDB_API_KEY already exported via /etc/rp_environment

OUTPUT_DIR=/root/runs/ctc1b_continued_v2
cd ~/omnilingual-asr
python -u -m workflows.recipes.wav2vec2.asr "$OUTPUT_DIR" --config-file /root/training/config_1b_continued_v2.yaml
