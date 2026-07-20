#!/bin/bash
# Launches the long single-cycle run: warm-started from step_6000, targeting
# ~90% of one full epoch (35_000 steps) in ONE continuous tri-stage anneal
# (rather than repeated short fresh-anneal chunks), with beta_language
# lowered to 0.2 to push more training exposure toward Kalenjin/Maasai (our
# two consistently worst-performing languages). See config_1b_v3.yaml.
set -e
source /etc/rp_environment
source ~/venv/bin/activate
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
# WANDB_API_KEY already exported via /etc/rp_environment

OUTPUT_DIR=/root/runs/ctc1b_v3
cd ~/omnilingual-asr
python -u -m workflows.recipes.wav2vec2.asr "$OUTPUT_DIR" --config-file /root/training/config_1b_v3.yaml
