import os, json, glob, random
from collections import defaultdict
import pyarrow.parquet as pq

OUT_ROOT = os.path.expanduser("~/data/processed")

# maps final competition language code -> source processed dirs that feed it
LANG_SOURCES = {
    "swa": ["swa"],
    "kik": ["kik"],
    "luo": ["luo"],
    "som": ["som_maxatire", "som_mogadishu"],
    "mas": ["mas"],
    "kln": ["kln"],
}

DEV_FRAC = 0.10
MIN_DEV_SPEAKERS = 12
SEED = 42

def build_split_for_lang(lang_code, source_dirs):
    by_speaker = defaultdict(int)  # speaker_id -> row count
    total_rows = 0
    for src in source_dirs:
        files = sorted(glob.glob(os.path.join(OUT_ROOT, src, "*.parquet")))
        for fp in files:
            spk_col = pq.read_table(fp, columns=["speaker_id"]).column("speaker_id").to_pylist()
            for spk in spk_col:
                by_speaker[spk or "__unknown__"] += 1
            total_rows += len(spk_col)

    speakers = list(by_speaker.keys())
    random.Random(SEED).shuffle(speakers)
    target = DEV_FRAC * total_rows
    dev_speakers, dev_n = set(), 0
    for spk in speakers:
        if dev_n >= target and len(dev_speakers) >= MIN_DEV_SPEAKERS:
            break
        dev_speakers.add(spk)
        dev_n += by_speaker[spk]

    print(f"[{lang_code}] total_rows={total_rows} unique_speakers={len(speakers)} "
          f"dev_speakers={len(dev_speakers)} dev_rows={dev_n} ({100*dev_n/total_rows:.1f}%)", flush=True)
    return {"total_rows": total_rows, "dev_rows": dev_n, "dev_speakers": sorted(dev_speakers)}

def main():
    out_dir = os.path.join(OUT_ROOT, "_splits")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for lang_code, source_dirs in LANG_SOURCES.items():
        result = build_split_for_lang(lang_code, source_dirs)
        with open(os.path.join(out_dir, f"{lang_code}_dev_speakers.json"), "w") as f:
            json.dump(result, f)
        summary[lang_code] = {"total_rows": result["total_rows"], "dev_rows": result["dev_rows"]}
    with open(os.path.join(out_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
