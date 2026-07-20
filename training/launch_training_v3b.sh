#!/bin/bash
# Resumes the long single-cycle run after it crashed at step 11_000 (a
# zero-length dev-split audio row broke periodic validation's WER
# calculation with an uncaught exception, killing the whole process).
# Warm-started from that run's step_11000 checkpoint, validation disabled
# entirely this time (see config_1b_v3.yaml regime section) so this exact
# crash can't recur. Separate OUTPUT_DIR and HF checkpoint prefix from the
# original v3 run so this run's step_1000, step_2000, ... don't collide with
# and overwrite the already-uploaded step_1000..step_11000 from before.
set -e
source /etc/rp_environment
source ~/venv/bin/activate
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
# WANDB_API_KEY already exported via /etc/rp_environment

OUTPUT_DIR=/root/runs/ctc1b_v3b
cd ~/omnilingual-asr
python -u -m workflows.recipes.wav2vec2.asr "$OUTPUT_DIR" --config-file /root/training/config_1b_v3.yaml
