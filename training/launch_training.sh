#!/bin/bash
# Launches the CTC-1B fine-tuning run.
#
# THE ONE THING THAT WILL QUIETLY WRECK THIS RUN IF CHANGED CARELESSLY:
# regime.num_steps in config_1b.yaml sets the LR scheduler's decay horizon
# (tri-stage: warmup -> hold -> decay). If num_steps is set much larger than
# what actually runs (e.g. relying on early-stop alone to decide when to
# stop), the LR never anneals and the model plateaus in a noisy band instead
# of converging. Keep num_steps a realistic budget; early-stop still works,
# it just won't get to run out the decay phase if it fires early.
set -e
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
# WANDB_API_KEY must already be exported in this shell / ~/.bashrc

OUTPUT_DIR=/root/runs/ctc1b
cd ~/omnilingual-asr
# -u: unbuffered stdout/stderr. Without this, redirecting to a log file switches Python
# from line-buffering to block-buffering, so progress can be genuinely happening with
# nothing appearing in the log for a long stretch -- indistinguishable from a hang.
python -u -m workflows.recipes.wav2vec2.asr "$OUTPUT_DIR" --config-file /root/training/config_1b.yaml
