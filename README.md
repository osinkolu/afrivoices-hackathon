# AfriVoices East Africa ASR Hackathon

Training code and pipeline for a single unified ASR model covering 6 East African
languages (Swahili, Kikuyu, Luo/Dholuo, Somali, Maasai, Kalenjin), built for the
AfriVoices East Africa ASR Hackathon (Kaggle, organized by Digital Umuganda +
Maseno University). Final result: **0.47017** mean WER on the leaderboard, 10th
place.

## Architecture

`omniASR_CTC_1B_v2` (Meta's Omnilingual ASR family) -- a wav2vec2-based CTC
encoder, ~1B parameters, character-level tokenizer (`omniASR_tokenizer_written_v2`,
vocab size 10,288). CTC was chosen over the LLM/seq2seq variant in the same
family for non-autoregressive, CPU-friendly inference and language-agnostic
`transcribe()` (no language-ID step needed at inference).

## Repo layout

- `data_processing/` -- pulls and converts the raw per-language source
  datasets (Swahili, Kikuyu, Luo, Somali x2 dialects, Maasai, Kalenjin) into a
  single partitioned-parquet format fairseq2's training recipe expects, plus
  a from-scratch speaker-disjoint train/dev split (the source repos' own
  splits were not speaker-disjoint). See `data_processing/README.md` for the
  full list of data-quality issues found and fixed along the way (base64-encoded
  audio rows, null-transcription rows, HF cache disk-leak, etc).
- `training/` -- the full training lineage, in order:
  - `config_1b.yaml` / `launch_training.sh` -- the original 8,000-step run.
  - `patch_omnilingual_asr_mojibake.py` -- the root-cause data fix (see below).
  - `config_1b_continued.yaml` / `launch_training_continued.sh` -- 2,000-step
    fresh-anneal warm-start after the mojibake fix.
  - `config_1b_continued_v2.yaml` / `launch_training_continued_v2.sh` --
    6,000-step follow-up.
  - `config_1b_v3.yaml` / `launch_training_v3.sh` / `launch_training_v3b.sh` --
    a long single-cycle ~35,000-step run (crashed once at step 11,000 on a
    dev-set zero-length-audio validation bug, resumed with validation disabled).
    This produced the final submitted checkpoint (`v3b/step_24000`).
  - `config_1b_v4.yaml` / `launch_training_v4.sh` -- a follow-up experiment
    pushing `beta_language` lower to further upweight Kalenjin/Maasai (tested,
    did not improve on step_24000 -- kept for the record).
  - `checkpoint_backup.py` -- watches for new checkpoints during training and
    backs each one up to a private HF model repo (a training pod can die;
    the checkpoints shouldn't die with it).
  - `evaluate.py` -- per-language WER/CER on the speaker-disjoint dev split,
    used to pick the best checkpoint (dev-set WER, not training loss).
  - `generate_submission.py` -- batched inference against the Kaggle test set.
  - `average_checkpoints.py` -- checkpoint-weight averaging (tested; a wash).
  - `inspect_predictions.py` / `measure_tag_impact.py` -- qualitative/quantitative
    error analysis tools used to understand *why* WER plateaued where it did
    (see Findings below).
- `colab_generate_submission.ipynb` -- a Colab fallback for generating the
  submission CSV when a training pod's GPU credits run out.

## Techniques used

- **Root-cause data fix**: found and fixed a mojibake/double-encoding bug in
  Kikuyu (dominant), Luo, and Somali training text -- UTF-8 diacritics
  (ĩ/ũ/Ĩ/Ũ) mis-decoded upstream as Latin-1/CP1252, producing
  2-character sequences the tokenizer couldn't represent (it emitted `<unk>`
  there, and the model learned to reproduce that at inference). Patched
  directly in the omnilingual-asr data-loading pipeline (`patch_omnilingual_asr_mojibake.py`)
  so training text is corrected before tokenization, rather than only
  post-processing predictions after the fact.
- **Speaker-disjoint dev split**, built from scratch for an honest
  model-selection signal (the source repos' own splits were not speaker-disjoint).
- **Language-balanced mixture sampling**: `beta_language` (sqrt-temperature-style
  weighting, `weight = (hours/total_hours)^beta`) tuned down to push more
  training exposure toward the lowest-resource languages (Kalenjin, Maasai)
  relative to their natural data volume.
- **One long continuous LR-anneal cycle** rather than many short fresh-restart
  cycles -- found repeated short warmup-hold-decay cycles show clearly
  diminishing returns compared to a single longer continuous schedule.
- **Dev-set-driven checkpoint selection**, not training-loss-driven --
  training loss and dev WER didn't always move together.
- **Checkpoint averaging** (tested; didn't help in this case, most likely
  because the averaged checkpoints were too close together / too correlated).
- No language-model decoding or data augmentation was used -- flagged as a
  lever we didn't have time to pursue.

## Compute

Single NVIDIA A40 (48GB) on RunPod, bf16 mixed precision, dynamic
length-based batching (~1.5M audio samples/step), gradient accumulation over
8 micro-batches. Roughly ~65-70 total GPU-hours across all training runs.
Earlier data-pipeline work (pulling/converting/resampling the raw source
datasets) ran on a separate CPU-only VM.

## Findings: why WER plateaued where it did

Direct inspection of real predictions vs. references (not just aggregate WER)
found two real but ultimately minor contributing artifacts:

1. Reference transcriptions contain literal transcriber annotation tags like
   `[cs]` (code-switch marker) and `[pause]` that the model correctly never
   produces -- these inflate WER by ~0.1-0.9 percentage points depending on
   language (worst for Maasai, which has them in ~13% of dev rows).
2. Kalenjin and Maasai lack a single standardized orthography in this
   dataset, so a portion of "errors" are phonetically-identical spelling
   variants (b/p, k/g, o/a swaps) rather than genuine recognition failures.

Neither fully explains the remaining gap to the leaderboard leader, though.
The dominant factors are most likely genuine training-data scarcity
(Kalenjin: 86k rows, Maasai: 53k rows, vs. Swahili's 503k -- a 6-9x gap even
after upweighting) and CTC's architectural ceiling without a language model
for decoding. Neither was fixable within the hackathon's remaining time/compute.
