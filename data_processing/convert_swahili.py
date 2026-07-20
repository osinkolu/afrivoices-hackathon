import os, io, gc, json, time, tarfile, shutil
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import librosa
import numpy as np
from pydub import AudioSegment
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ProcessPoolExecutor, as_completed

TOKEN = os.environ.get("HF_TOKEN")
OUT_ROOT = os.path.expanduser("~/data/processed")
TARGET_SR = 16000
# Swahili clips run 16-30s (vs ~1-3s for the Anv-ke languages), so a chunk with
# thousands of records held fully in memory before one big write can OOM even
# with plenty of total RAM once ~28 of these run concurrently (observed one
# worker alone hit 7.9GB RSS). Fewer workers + flushing writes incrementally
# (below) bounds memory instead of just reducing how often it's exercised.
NUM_WORKERS = max(1, (os.cpu_count() or 4) // 2)
FLUSH_EVERY = 150
REPO_ID = "DigitalUmuganda/Afrivoice_Swahili"
CATEGORIES = ["agriculture", "education", "financial", "government", "health"]
SPLITS = ["train", "dev", "test"]
LANG_CODE = "swa"

def convert_pydub_audio(raw_bytes, ext_hint):
    seg = AudioSegment.from_file(io.BytesIO(raw_bytes))
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)
    samples /= 32768.0
    if sr != TARGET_SR:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
    buf = io.BytesIO()
    sf.write(buf, samples, TARGET_SR, format="FLAC")
    return buf.getvalue(), len(samples) / TARGET_SR

def process_one_chunk(category, split, chunk_id, manifest_fname, audio_fname, out_dir):
    t0 = time.time()
    scratch_dir = os.path.join(OUT_ROOT, "_scratch", LANG_CODE)
    os.makedirs(scratch_dir, exist_ok=True)
    # local_dir downloads a real file (not a cache blob + symlink), so os.remove()
    # below actually reclaims disk instead of just deleting a symlink.
    manifest_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=manifest_fname, token=TOKEN,
                                     local_dir=scratch_dir)
    with open(manifest_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    audio_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=audio_fname, token=TOKEN,
                                  local_dir=scratch_dir)
    # .tar.xz is not seekable: extracting members out-of-order forces the LZMA
    # stream to re-decompress from the start each time (observed 4s+ per file).
    # Extract the whole archive once (single sequential decompress pass), then
    # read individual files back from disk, which is fast regardless of order.
    extract_dir = audio_path + "_extracted"
    with tarfile.open(audio_path, "r:*") as tf:
        tf.extractall(extract_dir, filter="data")
    path_by_basename = {}
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            path_by_basename[fn] = os.path.join(root, fn)

    schema = pa.schema({
        "audio_flac": pa.binary(), "text": pa.string(), "speaker_id": pa.string(),
        "split": pa.string(), "domain": pa.string(), "language": pa.string(),
    })
    out_name = f"{category}__{split}__{chunk_id}__proc.parquet"
    out_path = os.path.join(out_dir, out_name)

    buf_audio, buf_text, buf_speaker = [], [], []
    total_hours, skipped, kept = 0.0, 0, 0
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    try:
        def flush():
            if not buf_audio:
                return
            batch = pa.table({
                "audio_flac": buf_audio, "text": buf_text, "speaker_id": buf_speaker,
                "split": [split] * len(buf_audio), "domain": [category] * len(buf_audio),
                "language": [LANG_CODE] * len(buf_audio),
            }, schema=schema)
            writer.write_table(batch)
            buf_audio.clear(); buf_text.clear(); buf_speaker.clear()

        for rec in records:
            audio_fp = rec.get("audio_filepath")
            if not audio_fp:
                skipped += 1
                continue
            member_path = path_by_basename.get(os.path.basename(audio_fp))
            if member_path is None:
                skipped += 1
                continue
            try:
                with open(member_path, "rb") as f:
                    raw = f.read()
                flac_bytes, dur = convert_pydub_audio(raw, os.path.splitext(audio_fp)[1])
            except Exception:
                skipped += 1
                continue
            buf_audio.append(flac_bytes)
            buf_text.append(rec.get("transcription", ""))
            buf_speaker.append(rec.get("voice_creator_id", ""))
            total_hours += dur / 3600
            kept += 1
            if len(buf_audio) >= FLUSH_EVERY:
                flush()
        flush()
    finally:
        writer.close()

    shutil.rmtree(extract_dir, ignore_errors=True)
    for p in (manifest_path, audio_path):
        try:
            os.remove(p)
        except OSError:
            pass
    gc.collect()

    elapsed = time.time() - t0
    return {"src": f"{category}/{split}/{chunk_id}", "rows": len(records),
            "kept": kept, "skipped": skipped, "hours": total_hours, "elapsed": elapsed}

def list_chunks(api):
    info = api.dataset_info(REPO_ID, files_metadata=True)
    names = set(s.rfilename for s in info.siblings)
    chunks = []
    for category in CATEGORIES:
        for split in SPLITS:
            prefix = f"{category}_swahili_{split}"
            manifests = sorted(n for n in names if n.startswith(prefix + "/") and n.endswith(".jsonl"))
            for m in manifests:
                chunk_id = os.path.basename(m).split("_")[-1].split(".")[0]
                audio_fname = f"{prefix}/audio/audio_{chunk_id}.tar.xz"
                if audio_fname in names:
                    chunks.append((category, split, chunk_id, m, audio_fname))
                else:
                    print(f"WARN: no matching audio tar for {m}", flush=True)
    return chunks

def main(num_workers=NUM_WORKERS):
    api = HfApi(token=TOKEN)
    out_dir = os.path.join(OUT_ROOT, LANG_CODE)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "_progress.jsonl")

    done = set()
    cum_hours = 0.0
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec["src"])
                cum_hours += rec.get("hours", 0.0)

    chunks = list_chunks(api)
    pending = [c for c in chunks if f"{c[0]}/{c[1]}/{c[2]}" not in done]
    print(f"[swa] {len(done)} already done, {len(pending)} pending, using {num_workers} workers", flush=True)

    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex, open(log_path, "a", buffering=1) as logf:
        futures = {ex.submit(process_one_chunk, c[0], c[1], c[2], c[3], c[4], out_dir): c for c in pending}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[swa] FAILED {c[0]}/{c[1]}/{c[2]}: {e}", flush=True)
                continue
            cum_hours += res["hours"]
            completed += 1
            logf.write(json.dumps({"src": res["src"], "rows": res["rows"], "hours": res["hours"]}) + "\n")
            print(f"[swa] {completed}/{len(pending)} {res['src']} rows={res['rows']} "
                  f"kept={res['kept']} skipped={res['skipped']} cum_hours={cum_hours:.1f} "
                  f"took={res['elapsed']:.1f}s", flush=True)

if __name__ == "__main__":
    main()
