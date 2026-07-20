#!/bin/bash
# Run before launching training -- catches schema/path/tokenizer mistakes in seconds
# rather than after minutes of training startup.
set -e
cd ~/omnilingual-asr
python -m workflows.dataprep.dataloader_example \
  --dataset_path=/root/data/afrivoices_processed/version=0 --split=dev \
  --tokenizer_name=omniASR_tokenizer_written_v2 --num_iterations=5 --device=cpu
