# AfriVoices data processing pipeline

Downloads, cleans, and re-encodes the AfriVoices East Africa ASR competition data (6 languages,
7 source repos) into the partitioned-parquet format expected by Meta's `omnilingual-asr` training
recipe, with a speaker-disjoint dev split.

Ran on a CPU-only VM (32 vCPU / 128GB RAM / Ubuntu 24.04) — none of this needs a GPU. Output was
~204GB (16kHz FLAC) from ~1.37TB of raw source audio.

## Setup

```bash
pip install -r requirements.txt

# Swahili's audio is webm-encoded and needs ffmpeg (via pydub) to decode — a system binary,
# not just the pip package. Static build avoids relying on the OS package manager/mirrors:
curl -sS -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz
mkdir -p /tmp/ffmpeg_extract && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg_extract --strip-components=1
sudo mv /tmp/ffmpeg_extract/ffmpeg /tmp/ffmpeg_extract/ffprobe /usr/local/bin/

export HF_TOKEN=...       # needed for the Anv-ke / DigitalUmuganda repos
export HF_HUB_DISABLE_XET=1   # see "Known issues" below
```

## Pipeline (run in this order)

1. **`inspect_datasets.py`** — lists file counts/sizes for all 7 source repos before downloading
   anything. Useful for sanity-checking a new source before committing disk/bandwidth to it.

2. **`convert_anvke.py [lang_code]`** — downloads + converts the 5 `Anv-ke/*` repos (Kikuyu, Luo,
   Somali/Maxatire, Maasai, Kalenjin). Source audio is raw 44.1kHz mono WAV embedded in parquet.
   Resamples to 16kHz, FLAC-encodes, writes output parquet per source file, deletes the raw
   download. Parallelized across `cpu_count() - 4` worker processes. Omit `lang_code` to run all 5.

3. **`convert_swahili.py`** — downloads + converts `DigitalUmuganda/Afrivoice_Swahili`. This repo
   uses an older HF "loading script" format (manifests + `.tar.xz` audio archives + webm audio) —
   `datasets.load_dataset()` no longer supports this format as of `datasets>=4`, so this
   reimplements the extraction directly. Decodes via `pydub`/`ffmpeg` (webm), resamples, FLAC-encodes.

4. **`convert_somali_mogadishu.py`** — downloads + converts the `Somali/` subset of
   `DigitalUmuganda/Afrivoice` (a 1.2TB multilingual repo — only the Somali slice is pulled, via
   `allow_patterns`). Similar `.tar.xz`-archive format to Swahili, but raw WAV audio (no pydub
   needed). **~80% of records in this source have no transcription** (confirmed genuine — not a
   bug) and are skipped; expect far fewer usable hours than the raw file size suggests.

5. **`build_speaker_split.py`** — scans only the `speaker_id` column (no audio decode) across all
   processed output from steps 2-4, and for each of the 6 final competition languages, greedily
   assigns whole speakers to a dev split (target ~10% of rows, min 12 dev speakers) so dev speakers
   never appear in train. The two Somali sources (Maxatire + Mogadishu) are pooled under one `som`
   split. Writes `_splits/{lang}_dev_speakers.json` per language.

6. **`finalize_omni_format.py [lang1 lang2 ...]`** — rewrites everything into the final layout:
   `omni_final/version=0/corpus=afrivoices/split={train,dev}/language={code}/part-*.parquet`,
   applying the speaker split from step 5. Also computes `language_distribution_0.tsv` (hours per
   language, train split) for the recipe's temperature-sampling mixture dataloader. Omit language
   args to run all 6.

## Known issues found while building this (worth knowing before re-running)

- **`.tar.xz` is not seekable.** Extracting archive members out of manifest order forces the LZMA
  stream to re-decompress from the start each time — one measured case took 4+ seconds for a single
  file vs. milliseconds sequentially. Both `convert_swahili.py` and `convert_somali_mogadishu.py`
  extract the whole archive once up front instead of calling `extractfile()` per member.
- **Some Anv-ke rows store WAV audio as base64-encoded text**, not raw binary bytes (~3-4% of rows).
  `convert_anvke.py` and `convert_somali_mogadishu.py` detect this (missing `RIFF` header) and
  base64-decode as a fallback rather than silently dropping those rows.
- **Some source rows have a null (not missing) transcription.** `dict.get(key, default)` only
  substitutes `default` when the key is *absent*, not when its value is explicitly `None` — so rows
  with `"transcription": null` in the source slip through as `None` in `convert_anvke.py` and
  `convert_swahili.py` (found affecting ~4.5% of rows across all 6 languages once discovered).
  fairseq2's dataloader chokes on this (`TypeError: object of type 'float' has no len()`, since a
  null text becomes non-string by the time its `is_not_empty()` check runs `len(text)`).
  `finalize_omni_format.py` now filters any row where `text` isn't a non-empty string, since this is
  the one place all 6 languages funnel through regardless of source format.
- **`hf_hub_download()`'s default cache mode returns a symlink**, not a real file — `os.remove()` on
  it only deletes the symlink, leaving the actual multi-hundred-MB blob in `~/.cache/huggingface`.
  All scripts here pass `local_dir=` to force a real downloaded file so cleanup actually reclaims
  disk. (This caused a real incident: disk hit 982GB/991GB during the initial Anv-ke run before the
  fix — the processed output itself was only ever ~150-200GB.)
- **`HF_HUB_DISABLE_XET=1` is required.** The newer Xet transfer backend in `huggingface_hub`
  repeatedly stalled indefinitely on specific connections during bulk parallel downloads; plain
  HTTP was reliable throughout.
- **Memory**: Swahili/Somali clips run 16-30s vs. ~1-3s for the Anv-ke languages. Accumulating a
  whole chunk's decoded audio in memory before one big parquet write caused an OOM (one worker hit
  7.9GB RSS alone) once ~28 workers ran concurrently. `convert_swahili.py` and
  `convert_somali_mogadishu.py` flush to a `ParquetWriter` incrementally (every 150 rows) to bound
  memory regardless of chunk size, and use `cpu_count() // 2` workers rather than `cpu_count() - 4`.
- **A large fraction of Kikuyu's `text` has inherited double-encoding mojibake**, found *after*
  training (not caught during data prep — worth checking for in any re-run). Its two diacritics
  (`ĩ`, `ũ`, and capitalized `Ĩ`, `Ũ`) are two-byte UTF-8 characters; somewhere upstream (likely in
  the source repo itself, prior to this pipeline) those bytes got decoded as Latin-1 instead of
  UTF-8, producing a specific, reversible substitution:
  - `ĩ` (UTF-8 `C4 A9`) → `Ä©`　`ũ` (UTF-8 `C5 A9`) → `Å©`　`Ĩ` (UTF-8 `C4 A8`) → `Ä¨`　`Ũ` (UTF-8 `C5 A8`) → `Å¨`

  Scanning 25 Kikuyu train files found several **90-100% affected** (e.g. one file: 253/253 rows).
  Verified the reversal against real text: `"AndÅ© magÄ©thondeka tÅ©thandÅ©kÅ©"` → `"Andũ magĩthondeka
  tũthandũkũ"` ("People make small boxes..."), genuinely coherent Kikuyu, not noise. Since the ASR
  tokenizer doesn't have `©`/`¨` in its character vocabulary, it emits its UNK token (`U+2047`, `⁇`)
  at inference wherever the model learned this pattern from corrupted training text — `clean_pred()`
  in `training/generate_submission.py` and `training/evaluate.py` reconstructs the real character
  from context instead of just stripping UNK (a strictly worse mitigation that leaves a deletion
  error instead of recovering real text). Both `©` and `¨` collapse to the same UNK token, so
  upper/lowercase can't be perfectly recovered from model output alone; defaults to lowercase (the
  dominant case) except at sentence-initial position. **The real fix** would be re-encoding the
  affected source text correctly and retraining — not done here given deadline constraints.

## Output

Final dataset (~6,153 hours across `swa`/`kik`/`luo`/`som`/`mas`/`kln`) lives at
`~/data/processed/omni_final/` after step 6, and was pushed to a private HF dataset repo for the
training notebook to pull from directly (see the project's training notebook for that step).
