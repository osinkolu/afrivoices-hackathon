"""Runs the fine-tuned model against the official Kaggle test set and writes
submission.csv in the required id,language,transcription format.

CTC's transcribe() ignores the `language` argument, so one joint model call
handles every row regardless of language -- no per-language branching needed
(this also matches how a real deployment wouldn't reliably have language
metadata available).

Batches across multiple test rows per GPU call (not just chunks within a
single row) -- the earlier per-row version left the GPU mostly idle. Also
strips the tokenizer's UNK token (U+2047) from output text: a subset of
Kikuyu training data has inherited double-encoding mojibake (diacritics like
ĩ/ũ corrupted into e.g. "Ä©"/"Å©"), and since "©" isn't in the ASR tokenizer's
character vocabulary, the model learned to emit UNK there. Not fixable
without retraining on re-encoded data; stripping it from predictions is the
practical mitigation for a submission.

Usage: python generate_submission.py /root/runs/ctc1b/ws_1.../checkpoints/step_N/model/pp_00/tp_00/sdp_00.pt
"""
import sys, os, io, glob, csv, base64, re, unicodedata
from collections import defaultdict
import torch
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import kagglehub

from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

SR = 16000
# Rows can be up to ~39.5s each; batch memory scales with batch_size x max
# length in the batch, which varies unpredictably (test clips aren't uniform
# length) -- both 32 and 20 eventually hit a long-clip batch and OOM'd, each
# costing a restart. 8 ran cleanly through many files without failing; given
# repeated crashes cost more time than the extra speedup would save, this
# prioritizes finishing reliably over maximizing GPU%.
BATCH_ROWS = 8
# Kaggle's grader reads the CSV (almost certainly via pandas) and rejects the
# whole submission if any cell is null -- an empty string gets coerced to NaN
# on read. Every row must have SOME non-empty text, even when transcription
# failed entirely or the cleaned prediction happened to be empty.
EMPTY_PLACEHOLDER = "."


def clean_pred(t):
    # A subset of Kikuyu training data has inherited double-encoding mojibake:
    # its two diacritics were UTF-8, but got decoded as Latin-1 somewhere
    # upstream, producing "A"+combining-char sequences the ASR tokenizer
    # doesn't have in vocab -- so it emits UNK (U+2047) there instead:
    #   ĩ (UTF-8 C4 A9) -> misread as "Ä©" -> tokenizer sees "Ä" + UNK
    #   ũ (UTF-8 C5 A9) -> misread as "Å©" -> tokenizer sees "Å" + UNK
    #   Ĩ (UTF-8 C4 A8) -> misread as "Ä¨" -> tokenizer sees "Ä" + UNK
    #   Ũ (UTF-8 C5 A8) -> misread as "Å¨" -> tokenizer sees "Å" + UNK
    # Verified against real training text: "AndÅ© magÄ©thondeka" reverses to
    # "Andũ magĩthondeka" ("People make..."), genuinely coherent Kikuyu, not
    # noise. Since "©" and "¨" both collapse to the same UNK token, upper vs.
    # lower case can't be perfectly recovered from output alone -- default to
    # lowercase (the dominant case) except at sentence-initial position.
    #
    # Scanning all 6 languages' training text for similar patterns found two
    # more, both much smaller in scale than Kikuyu's:
    # - Somali: some captions contain decorative "bold" Unicode letters (e.g.
    #   the U+1D400 mathematical alphanumeric block) copy-pasted from social
    #   media text, which the tokenizer also can't represent. NFKC
    #   normalization maps these back to plain letters -- safe globally, since
    #   it only touches compatibility variants, not genuine diacritics like ĩ/ũ.
    # - Luo: a much smaller-scale (~0.1% of sampled chars) case where a
    #   right-quote/apostrophe got double-encoded via Windows-1252 (not plain
    #   Latin-1) into "â€™" (bytes E2 80 99 misread) -- reversed here too.
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("â€™", "'")

    def repl(m, upper, lower):
        prefix = t[:m.start()]
        return upper if (m.start() == 0 or re.search(r"[.!?]\s*$", prefix)) else lower

    t = re.sub(r"Å\s*⁇\s*", lambda m: repl(m, "Ũ", "ũ"), t)
    t = re.sub(r"Ä\s*⁇\s*", lambda m: repl(m, "Ĩ", "ĩ"), t)
    t = t.replace("⁇", "")  # strip any remaining/different UNK occurrences
    return re.sub(r"\s+", " ", t).strip()


def read_audio_bytes(raw_bytes):
    # Same base64-encoding quirk found in the training source data (see
    # data_processing/README.md) -- some rows here store the WAV payload as
    # base64 text instead of raw binary. Try raw first, fall back to decoding.
    if not raw_bytes.startswith(b"RIFF"):
        try:
            raw_bytes = base64.b64decode(raw_bytes)
        except Exception:
            pass
    return sf.read(io.BytesIO(raw_bytes))


def prep(wav):
    # pad <2s (bare CTC forward pass on a very short clip can collapse to empty
    # output), split >~39.5s into <=35s chunks (inference limit)
    if len(wav) < 2 * SR:
        wav = np.concatenate([wav, np.zeros(2 * SR - len(wav), np.float32)])
    n = 35 * SR
    return [wav] if len(wav) <= int(39.5 * SR) else [wav[i:i + n] for i in range(0, len(wav), n)]


def load_finetuned(ckpt_pt, device="cuda", dtype=torch.bfloat16):
    model = load_model("omniASR_CTC_1B_v2", device=torch.device(device), dtype=dtype)
    sd = torch.load(ckpt_pt, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def flush_batch(buffer, pipe, writer):
    """buffer: list of (id, lang, chunks). Transcribes all chunks from all
    rows in one batched call, then regroups per-row and writes."""
    all_chunks, owner = [], []
    for i, (_id, lang, chunks) in enumerate(buffer):
        for c in chunks:
            all_chunks.append(c)
            owner.append(i)

    texts = pipe.transcribe(
        [{"waveform": torch.from_numpy(c), "sample_rate": SR} for c in all_chunks],
        batch_size=len(all_chunks),
    )
    grouped = defaultdict(list)
    for idx, txt in zip(owner, texts):
        grouped[idx].append(txt)

    for i, (_id, lang, _chunks) in enumerate(buffer):
        prediction = clean_pred(" ".join(grouped.get(i, []))) or EMPTY_PLACEHOLDER
        writer.writerow({"id": _id, "language": lang, "transcription": prediction})


def main(ckpt_path, out_csv="submission.csv"):
    test_path = kagglehub.dataset_download("digitalumuganda/anv-test-data-nt")
    print("test set at:", test_path, flush=True)

    done_ids = set()
    if os.path.exists(out_csv):
        with open(out_csv, encoding="utf-8") as f:
            done_ids = {r["id"] for r in csv.DictReader(f)}
        print(f"resuming: {len(done_ids)} ids already in {out_csv}", flush=True)

    model = load_finetuned(ckpt_path)
    pipe = ASRInferencePipeline(model_card=None, model=model,
                                 tokenizer=load_tokenizer("omniASR_tokenizer_written_v2"))

    files = sorted(glob.glob(f"{test_path}/*/*/*.parquet"))
    total_rows, failed_rows = len(done_ids), 0

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "language", "transcription"])
        if write_header:
            w.writeheader()

        buffer = []
        for i, fp in enumerate(files):
            t = pq.read_table(fp, columns=["id", "audio", "language"])
            ids = t.column("id").to_pylist()
            audios = t.column("audio").to_pylist()
            langs = t.column("language").to_pylist()
            for _id, a, lang in zip(ids, audios, langs):
                if _id in done_ids:
                    continue
                raw = a["bytes"] if isinstance(a, dict) else a
                try:
                    wav, sr = read_audio_bytes(raw)
                    if sr != SR:
                        import librosa
                        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
                    chunks = prep(wav.astype(np.float32))
                except Exception as e:
                    failed_rows += 1
                    print(f"WARN: failed on id={_id} lang={lang}: {e}", flush=True)
                    # still emit a row -- every test id must appear or row-count
                    # validation against the official test set will fail. Never
                    # write an empty prediction (see EMPTY_PLACEHOLDER above).
                    w.writerow({"id": _id, "language": lang, "transcription": EMPTY_PLACEHOLDER})
                    total_rows += 1
                    continue

                buffer.append((_id, lang, chunks))
                total_rows += 1
                if len(buffer) >= BATCH_ROWS:
                    flush_batch(buffer, pipe, w)
                    buffer = []

            if buffer:
                flush_batch(buffer, pipe, w)
                buffer = []
            f.flush()
            print(f"{i+1}/{len(files)} files done, {total_rows} rows so far ({failed_rows} failed)", flush=True)

    print(f"wrote {out_csv}, total {total_rows} rows ({failed_rows} failed)", flush=True)


if __name__ == "__main__":
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "submission.csv"
    main(ckpt, out)
