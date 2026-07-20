import os, io, sys, gc, json, time, base64
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import librosa
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ProcessPoolExecutor, as_completed

TOKEN = os.environ.get("HF_TOKEN")
OUT_ROOT = os.path.expanduser("~/data/processed")
TARGET_SR = 16000
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 4)

LANG_REPOS = {
    "kik": "Anv-ke/kikuyu",
    "luo": "Anv-ke/Dholuo",
    "som_maxatire": "Anv-ke/Somali",
    "mas": "Anv-ke/Maasai",
    "kln": "Anv-ke/Kalenjin",
}

def _read_audio_bytes(raw_bytes):
    # Some rows in these repos store the WAV payload as a base64-encoded
    # string cast to bytes, instead of the raw binary (~3-4% of rows observed).
    # Raw binary WAV starts with b"RIFF"; anything else, try base64-decoding first.
    if not raw_bytes.startswith(b"RIFF"):
        try:
            raw_bytes = base64.b64decode(raw_bytes)
        except Exception:
            pass
    return sf.read(io.BytesIO(raw_bytes))

def convert_row_audio(raw_bytes):
    wav, sr = _read_audio_bytes(raw_bytes)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    buf = io.BytesIO()
    sf.write(buf, wav, TARGET_SR, format="FLAC")
    return buf.getvalue(), len(wav) / TARGET_SR

def process_one_file(repo_id, lang_code, fname, out_dir):
    t0 = time.time()
    scratch_dir = os.path.join(OUT_ROOT, "_scratch", lang_code)
    os.makedirs(scratch_dir, exist_ok=True)
    # local_dir downloads a real file (not a cache blob + symlink), so os.remove()
    # below actually reclaims disk instead of just deleting a symlink.
    local_path = hf_hub_download(repo_id, repo_type="dataset", filename=fname, token=TOKEN,
                                  local_dir=scratch_dir)
    table = pq.read_table(local_path)
    cols = table.column_names
    audio_col = "audio" if "audio" in cols else "audio_bytes"

    audio_list = table.column(audio_col).to_pylist()
    text_list = table.column("transcription").to_pylist() if "transcription" in cols else [""] * table.num_rows
    speaker_list = (table.column("recorder_uuid").to_pylist() if "recorder_uuid" in cols
                     else table.column("speaker_id").to_pylist() if "speaker_id" in cols
                     else [""] * table.num_rows)
    split_list = table.column("split").to_pylist() if "split" in cols else [""] * table.num_rows
    domain_list = table.column("domain").to_pylist() if "domain" in cols else [""] * table.num_rows

    new_audio, new_text, new_speaker, new_split, new_domain = [], [], [], [], []
    n = table.num_rows
    total_hours = 0.0
    skipped = 0
    for r in range(n):
        a = audio_list[r]
        raw = a["bytes"] if isinstance(a, dict) else a
        try:
            flac_bytes, dur = convert_row_audio(raw)
        except Exception as e:
            skipped += 1
            continue
        new_audio.append(flac_bytes)
        new_text.append(text_list[r])
        new_speaker.append(speaker_list[r])
        new_split.append(split_list[r])
        new_domain.append(domain_list[r])
        total_hours += dur / 3600

    out_table = pa.table({
        "audio_flac": new_audio,
        "text": new_text,
        "speaker_id": new_speaker,
        "split": new_split,
        "domain": new_domain,
        "language": [lang_code] * len(new_audio),
    })
    out_name = fname.replace("/", "__").replace(".parquet", "") + "__proc.parquet"
    pq.write_table(out_table, os.path.join(out_dir, out_name), compression="zstd")

    try:
        os.remove(local_path)
    except OSError:
        pass
    del table
    gc.collect()

    elapsed = time.time() - t0
    return {"src": fname, "rows": n, "kept": len(new_audio), "skipped": skipped,
            "hours": total_hours, "elapsed": elapsed}

def process_repo(lang_code, repo_id, num_workers=NUM_WORKERS):
    api = HfApi(token=TOKEN)
    info = api.dataset_info(repo_id, files_metadata=True)
    parquet_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")]
    out_dir = os.path.join(OUT_ROOT, lang_code)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "_progress.jsonl")
    done_files = set()
    cum_hours = 0.0
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                done_files.add(rec["src"])
                cum_hours += rec.get("hours", 0.0)

    pending = [f for f in parquet_files if f not in done_files]
    print(f"[{lang_code}] {len(done_files)} already done, {len(pending)} pending, "
          f"using {num_workers} workers", flush=True)

    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex, open(log_path, "a", buffering=1) as logf:
        futures = {ex.submit(process_one_file, repo_id, lang_code, fname, out_dir): fname
                   for fname in pending}
        for fut in as_completed(futures):
            fname = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[{lang_code}] FAILED {fname}: {e}", flush=True)
                continue
            cum_hours += res["hours"]
            completed += 1
            logf.write(json.dumps({"src": res["src"], "rows": res["rows"], "hours": res["hours"]}) + "\n")
            print(f"[{lang_code}] {completed}/{len(pending)} {fname} rows={res['rows']} "
                  f"kept={res['kept']} skipped={res['skipped']} cum_hours={cum_hours:.1f} "
                  f"took={res['elapsed']:.1f}s", flush=True)

if __name__ == "__main__":
    lang_arg = sys.argv[1] if len(sys.argv) > 1 else None
    targets = {lang_arg: LANG_REPOS[lang_arg]} if lang_arg else LANG_REPOS
    for lang_code, repo_id in targets.items():
        print(f"=== Processing {lang_code} ({repo_id}) ===", flush=True)
        process_repo(lang_code, repo_id)
