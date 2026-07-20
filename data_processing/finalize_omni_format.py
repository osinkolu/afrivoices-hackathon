import os, io, json, glob, time
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT_ROOT = os.path.expanduser("~/data/processed")
FINAL_ROOT = os.path.join(OUT_ROOT, "omni_final", "version=0", "corpus=afrivoices")
NUM_WORKERS = max(1, (os.cpu_count() or 4) // 2)
BATCH_SIZE = 200

LANG_SOURCES = {
    "swa": ["swa"],
    "kik": ["kik"],
    "luo": ["luo"],
    "som": ["som_maxatire", "som_mogadishu"],
    "mas": ["mas"],
    "kln": ["kln"],
}

SCHEMA = pa.schema({"text": pa.string(), "audio_bytes": pa.binary(), "audio_size": pa.int64()})

def process_one_file(lang_code, src_dir, fp, dev_speakers_set, part_tag):
    t0 = time.time()
    writers = {}  # split -> ParquetWriter
    out_paths = {}
    counts = {"train": 0, "dev": 0}
    hours = {"train": 0.0, "dev": 0.0}

    skipped_bad_text = 0
    pf = pq.ParquetFile(fp)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["audio_flac", "text", "speaker_id"]):
        d = batch.to_pydict()
        buf = {"train": {"text": [], "audio_bytes": [], "audio_size": []},
               "dev": {"text": [], "audio_bytes": [], "audio_size": []}}
        for flac_bytes, text, spk in zip(d["audio_flac"], d["text"], d["speaker_id"]):
            # Some source rows have a null/missing transcription (dict.get(key, default)
            # only substitutes default when the KEY is absent, not when its value is
            # explicitly None -- so rows with `"transcription": null` in the source slip
            # through as None here). A null text becomes a non-string (often float NaN)
            # by the time fairseq2's dataloader reads it, and its is_not_empty() check
            # does len(text), which throws on a float. Drop these rows -- they have no
            # usable label anyway.
            if not isinstance(text, str) or len(text) == 0:
                skipped_bad_text += 1
                continue
            split = "dev" if (spk or "__unknown__") in dev_speakers_set else "train"
            info = sf.info(io.BytesIO(flac_bytes))
            n_samples = int(info.frames)
            buf[split]["text"].append(text)
            buf[split]["audio_bytes"].append(flac_bytes)
            buf[split]["audio_size"].append(n_samples)
            counts[split] += 1
            hours[split] += n_samples / info.samplerate / 3600

        for split, cols in buf.items():
            if not cols["text"]:
                continue
            if split not in writers:
                out_dir = os.path.join(FINAL_ROOT, f"split={split}", f"language={lang_code}")
                os.makedirs(out_dir, exist_ok=True)
                out_paths[split] = os.path.join(out_dir, f"part-{src_dir}-{part_tag}.parquet")
                writers[split] = pq.ParquetWriter(out_paths[split], SCHEMA, compression="zstd")
            writers[split].write_table(pa.table(cols, schema=SCHEMA))

    for w in writers.values():
        w.close()

    return {"src": f"{src_dir}/{part_tag}", "counts": counts, "hours": hours,
            "skipped_bad_text": skipped_bad_text, "elapsed": time.time() - t0}

def main(langs=None):
    targets = {k: v for k, v in LANG_SOURCES.items() if langs is None or k in langs}
    jobs = []
    for lang_code, source_dirs in targets.items():
        split_info = json.load(open(os.path.join(OUT_ROOT, "_splits", f"{lang_code}_dev_speakers.json")))
        dev_set = set(split_info["dev_speakers"])
        for src_dir in source_dirs:
            files = sorted(glob.glob(os.path.join(OUT_ROOT, src_dir, "*.parquet")))
            for fp in files:
                part_tag = os.path.splitext(os.path.basename(fp))[0]
                jobs.append((lang_code, src_dir, fp, dev_set, part_tag))

    print(f"total jobs: {len(jobs)}, using {NUM_WORKERS} workers", flush=True)
    lang_totals = {l: {"train": 0.0, "dev": 0.0} for l in targets}
    completed = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(process_one_file, *job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            lang_code = job[0]
            try:
                res = fut.result()
            except Exception as e:
                print(f"FAILED {job[0]}/{job[1]}/{job[4]}: {e}", flush=True)
                continue
            completed += 1
            lang_totals[lang_code]["train"] += res["hours"]["train"]
            lang_totals[lang_code]["dev"] += res["hours"]["dev"]
            print(f"{completed}/{len(jobs)} {res['src']} lang={lang_code} "
                  f"train_h={lang_totals[lang_code]['train']:.1f} dev_h={lang_totals[lang_code]['dev']:.1f} "
                  f"skipped_bad_text={res['skipped_bad_text']} took={res['elapsed']:.1f}s", flush=True)

    print("=== FINAL TOTALS ===", flush=True)
    for l, v in lang_totals.items():
        print(f"{l}: train={v['train']:.1f}h dev={v['dev']:.1f}h total={v['train']+v['dev']:.1f}h", flush=True)

    with open(os.path.join(OUT_ROOT, "_splits", "final_hours.json"), "w") as f:
        json.dump(lang_totals, f, indent=2)

if __name__ == "__main__":
    import sys
    langs = sys.argv[1:] or None
    main(langs)
