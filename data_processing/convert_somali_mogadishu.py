import os, io, gc, json, time, tarfile, shutil, base64
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
NUM_WORKERS = max(1, (os.cpu_count() or 4) // 2)
FLUSH_EVERY = 150
REPO_ID = "DigitalUmuganda/Afrivoice"
LANG_CODE = "som_mogadishu"

def _read_audio_bytes(raw_bytes):
    # Same base64-encoding quirk seen in the Anv-ke repos; cheap to guard against here too.
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

def process_one_shard(shard_id, manifest_fname, audio_fname, out_dir):
    t0 = time.time()
    scratch_dir = os.path.join(OUT_ROOT, "_scratch", LANG_CODE)
    os.makedirs(scratch_dir, exist_ok=True)

    manifest_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=manifest_fname, token=TOKEN,
                                     local_dir=scratch_dir)
    with open(manifest_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    audio_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=audio_fname, token=TOKEN,
                                  local_dir=scratch_dir)
    # .tar.xz isn't seekable: extract once (sequential decompress) rather than
    # calling extractfile() per member out of manifest order (see Swahili fix).
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
    out_name = f"shard_{shard_id}__proc.parquet"
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
                "split": ["train"] * len(buf_audio), "domain": ["mogadishu"] * len(buf_audio),
                "language": [LANG_CODE] * len(buf_audio),
            }, schema=schema)
            writer.write_table(batch)
            buf_audio.clear(); buf_text.clear(); buf_speaker.clear()

        for rec in records:
            text = rec.get("transcription")
            audio_fp = rec.get("audio_filepath")
            if not text or not audio_fp:
                skipped += 1
                continue
            member_path = path_by_basename.get(os.path.basename(audio_fp))
            if member_path is None:
                skipped += 1
                continue
            try:
                with open(member_path, "rb") as f:
                    raw = f.read()
                flac_bytes, dur = convert_row_audio(raw)
            except Exception:
                skipped += 1
                continue
            buf_audio.append(flac_bytes)
            buf_text.append(text)
            buf_speaker.append(rec.get("speaker_id", ""))
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
    return {"src": f"shard_{shard_id}", "rows": len(records),
            "kept": kept, "skipped": skipped, "hours": total_hours, "elapsed": elapsed}

def list_shards(api):
    info = api.dataset_info(REPO_ID, files_metadata=True)
    names = set(s.rfilename for s in info.siblings if s.rfilename.startswith("Somali/"))
    manifests = sorted(n for n in names if n.startswith("Somali/manifest_") and n.endswith(".json"))
    shards = []
    for m in manifests:
        shard_id = os.path.basename(m).split("_")[-1].split(".")[0]
        audio_fname = f"Somali/audio_shards/audio_{shard_id}.tar.xz"
        if audio_fname in names:
            shards.append((shard_id, m, audio_fname))
        else:
            print(f"WARN: no matching audio tar for {m}", flush=True)
    return shards

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

    shards = list_shards(api)
    pending = [s for s in shards if f"shard_{s[0]}" not in done]
    print(f"[{LANG_CODE}] {len(done)} already done, {len(pending)} pending, using {num_workers} workers", flush=True)

    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex, open(log_path, "a", buffering=1) as logf:
        futures = {ex.submit(process_one_shard, s[0], s[1], s[2], out_dir): s for s in pending}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[{LANG_CODE}] FAILED shard_{s[0]}: {e}", flush=True)
                continue
            cum_hours += res["hours"]
            completed += 1
            logf.write(json.dumps({"src": res["src"], "rows": res["rows"], "hours": res["hours"]}) + "\n")
            print(f"[{LANG_CODE}] {completed}/{len(pending)} {res['src']} rows={res['rows']} "
                  f"kept={res['kept']} skipped={res['skipped']} cum_hours={cum_hours:.1f} "
                  f"took={res['elapsed']:.1f}s", flush=True)

if __name__ == "__main__":
    main()
