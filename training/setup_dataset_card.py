"""Writes the omnilingual-asr dataset asset card pointing at the local processed data.

Run once after pulling Professor/afrivoices-processed onto the training machine.
"""
import os

OMNI_REPO = os.path.expanduser("~/omnilingual-asr")
DATA_ROOT = os.path.expanduser("~/data/afrivoices_processed")

card_dir = os.path.join(OMNI_REPO, "src/omnilingual_asr/cards/datasets")
os.makedirs(card_dir, exist_ok=True)

card_path = os.path.join(card_dir, "afrivoices_asr.yaml")
with open(card_path, "w") as f:
    f.write(f"""
name: afrivoices_asr
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: {DATA_ROOT}/version=0
tokenizer_ref: omniASR_tokenizer_written_v2
""")

print(f"wrote {card_path}")
with open(card_path) as f:
    print(f.read())
