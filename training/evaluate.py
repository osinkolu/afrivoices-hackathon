"""Per-language WER/CER evaluation on the speaker-disjoint dev split.

Trust these numbers for model selection -- the dev split was built specifically
to be speaker-disjoint from train (the source repos' own splits were not, when
checked during data prep), so this is an honest proxy for unseen-speaker
performance rather than a memorized-speaker score.

Batches across multiple dev rows per GPU call (same approach as
generate_submission.py) -- the original row-at-a-time version was fine for
quick spot checks, but the actual dev split turned out to be much larger than
expected (113k rows across 6 languages), which would take many hours
unbatched. Also caps rows evaluated per language (see MAX_ROWS_PER_LANG) so a
full 2-checkpoint comparison finishes in a practical time; this is a sampled
estimate, not the full-dev-set number, and is logged as such below.

Usage: python evaluate.py /root/checkpoints/step_N/sdp_00.pt
"""
import sys, os, io, re, glob, unicodedata
from collections import defaultdict
import torch, jiwer
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np

from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

DATA_ROOT = "/root/data/afrivoices_processed/version=0/corpus=afrivoices/split=dev"
SR = 16000
PUNCT_RE = re.compile(r"[.,?!;:\"'()\[\]{}\\/—\-]")
BATCH_ROWS = int(os.environ.get("BATCH_ROWS", "8"))  # lower via env var when running
# alongside a concurrent training job on the same GPU (shares VRAM headroom)
# Full dev split is 113k rows across 6 languages -- unbatched or not, scoring
# all of it per checkpoint isn't a practical use of time when comparing
# candidates. Every Nth row (not just the first N) keeps the sample spread
# across all source files rather than biased toward whatever sorts first.
MAX_ROWS_PER_LANG = int(os.environ.get("MAX_ROWS_PER_LANG", "1000"))


def normalize(s):
    return re.sub(r"\s+", " ", PUNCT_RE.sub(" ", (s or "").lower())).strip()


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
    # noise. Reconstructing this (rather than just stripping UNK) recovers a
    # real character instead of leaving a deletion error, which is a much
    # better mitigation for scoring/submission than blind stripping. Since "©"
    # and "¨" both collapse to the same UNK token, upper vs. lower case can't
    # be perfectly recovered from output alone -- default to lowercase (the
    # dominant case) except at sentence-initial position.
    #
    # Note: the training-time fix (patch_omnilingual_asr_mojibake.py) now
    # reconstructs this exactly (no case ambiguity) before the model ever
    # sees it, so a model trained on the corrected pipeline shouldn't produce
    # this pattern at all -- this is now a no-op for that model's output, and
    # only still matters for scoring the old (pre-fix) checkpoint fairly.
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
    t = unicodedata.normalize("NFKC", t or "")
    t = t.replace("â€™", "'")

    def repl(m, upper, lower):
        prefix = t[:m.start()]
        return upper if (m.start() == 0 or re.search(r"[.!?]\s*$", prefix)) else lower

    t = re.sub(r"Å\s*⁇\s*", lambda m: repl(m, "Ũ", "ũ"), t)
    t = re.sub(r"Ä\s*⁇\s*", lambda m: repl(m, "Ĩ", "ĩ"), t)
    t = t.replace("⁇", "")  # strip any remaining/different UNK occurrences
    return re.sub(r"\s+", " ", t).strip()


def prep(wav):
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


def score(refs, hyps):
    out = {}
    for proto, tx in (("raw", lambda x: x), ("norm", normalize)):
        R, H = [tx(r) for r in refs], [tx(h) for h in hyps]
        out[proto] = dict(wer=jiwer.wer(R, H), cer=jiwer.cer(R, H))
    return out


def flush_batch(buffer, pipe, refs, hyps):
    """buffer: list of (ref_text, chunks). Transcribes all chunks from all
    rows in one batched call, then regroups per-row."""
    all_chunks, owner = [], []
    for i, (_ref, chunks) in enumerate(buffer):
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

    for i, (ref, _chunks) in enumerate(buffer):
        refs.append(ref)
        hyps.append(clean_pred(" ".join(grouped.get(i, []))))


def iter_sampled_rows(lang_dir, max_rows):
    """Yields (ref_text, audio_bytes) pairs, spread across all files in the
    language's dev dir rather than just the first files encountered, and
    skipping the null (NaN/float) transcription rows present in a subset of
    the source data (see data_processing/README.md)."""
    files = sorted(glob.glob(os.path.join(lang_dir, "*.parquet")))
    total = 0
    for fp in files:
        total += pq.read_metadata(fp).num_rows
    stride = max(1, total // max_rows) if max_rows > 0 else 1

    seen = 0
    yielded = 0
    for fp in files:
        t = pq.read_table(fp, columns=["audio_bytes", "text"])
        for raw, ref_text in zip(t.column("audio_bytes").to_pylist(), t.column("text").to_pylist()):
            if not (isinstance(ref_text, str) and len(ref_text) > 0):
                seen += 1
                continue
            if seen % stride == 0:
                if max_rows > 0 and yielded >= max_rows:
                    return
                yield ref_text, raw
                yielded += 1
            seen += 1


def main(ckpt_path):
    model = load_finetuned(ckpt_path)
    pipe = ASRInferencePipeline(model_card=None, model=model,
                                 tokenizer=load_tokenizer("omniASR_tokenizer_written_v2"))

    per_lang_results = {}
    for lang_dir in sorted(glob.glob(os.path.join(DATA_ROOT, "language=*"))):
        lang = os.path.basename(lang_dir).split("=")[1]
        refs, hyps = [], []
        buffer = []
        for ref_text, raw in iter_sampled_rows(lang_dir, MAX_ROWS_PER_LANG):
            wav, sr = sf.read(io.BytesIO(raw))
            chunks = prep(wav.astype(np.float32))
            buffer.append((ref_text, chunks))
            if len(buffer) >= BATCH_ROWS:
                flush_batch(buffer, pipe, refs, hyps)
                buffer = []
        if buffer:
            flush_batch(buffer, pipe, refs, hyps)

        result = score(refs, hyps)
        per_lang_results[lang] = result
        print(f"[{lang}] n={len(refs)} (sampled, stride-based) raw_wer={result['raw']['wer']:.4f} "
              f"raw_cer={result['raw']['cer']:.4f} norm_wer={result['norm']['wer']:.4f} "
              f"norm_cer={result['norm']['cer']:.4f}", flush=True)

    # Leaderboard metric is the UNWEIGHTED MEAN of per-language WER -- a neglected
    # language drags the average down as much as the best one pulls it up.
    mean_raw_wer = sum(r["raw"]["wer"] for r in per_lang_results.values()) / len(per_lang_results)
    mean_norm_wer = sum(r["norm"]["wer"] for r in per_lang_results.values()) / len(per_lang_results)
    print(f"\n=== unweighted mean WER across {len(per_lang_results)} languages "
          f"(capped at {MAX_ROWS_PER_LANG} sampled rows/language): "
          f"raw={mean_raw_wer:.4f} norm={mean_norm_wer:.4f} ===")


if __name__ == "__main__":
    main(sys.argv[1])
