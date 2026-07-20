"""Qualitative error inspection: prints reference vs. prediction side by side
for a handful of dev-set examples per language, plus a word-level diff, so we
can see what KIND of errors the model is actually making (real semantic
mistakes vs. surface-level punctuation/spacing noise vs. reference-side
transcription quality issues) -- aggregate WER numbers alone don't distinguish
these.

Usage: python inspect_predictions.py CKPT.pt [--n N] [--lang LANG]
"""
import sys, os, io, argparse, glob, difflib
import torch
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import clean_pred, prep, load_finetuned, DATA_ROOT, SR


def word_diff(ref, hyp):
    ref_words, hyp_words = ref.split(), hyp.split()
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        r = " ".join(ref_words[i1:i2]) or "-"
        h = " ".join(hyp_words[j1:j2]) or "-"
        out.append(f"    [{tag}] ref:{r!r} -> hyp:{h!r}")
    return out


def main(ckpt_path, n_per_lang, only_lang):
    model = load_finetuned(ckpt_path)
    pipe = ASRInferencePipeline(model_card=None, model=model,
                                 tokenizer=load_tokenizer("omniASR_tokenizer_written_v2"))

    for lang_dir in sorted(glob.glob(os.path.join(DATA_ROOT, "language=*"))):
        lang = os.path.basename(lang_dir).split("=")[1]
        if only_lang and lang != only_lang:
            continue
        print(f"\n{'='*70}\n{lang}\n{'='*70}")
        shown = 0
        for fp in sorted(glob.glob(os.path.join(lang_dir, "*.parquet"))):
            if shown >= n_per_lang:
                break
            t = pq.read_table(fp, columns=["audio_bytes", "text"])
            for raw, ref_text in zip(t.column("audio_bytes").to_pylist(), t.column("text").to_pylist()):
                if shown >= n_per_lang:
                    break
                if not (isinstance(ref_text, str) and len(ref_text) > 0):
                    continue
                wav, sr = sf.read(io.BytesIO(raw))
                chunks = prep(wav.astype(np.float32))
                texts = pipe.transcribe(
                    [{"waveform": torch.from_numpy(c), "sample_rate": sr} for c in chunks],
                    batch_size=16,
                )
                hyp = clean_pred(" ".join(texts))
                print(f"\n--- example {shown+1} ({len(wav)/sr:.1f}s) ---")
                print(f"  REF: {ref_text}")
                print(f"  HYP: {hyp}")
                diffs = word_diff(ref_text.lower(), hyp.lower())
                if diffs:
                    print("  DIFF:")
                    for d in diffs:
                        print(d)
                else:
                    print("  DIFF: (exact match, case/punct-insensitive)")
                shown += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--lang", default=None)
    args = ap.parse_args()
    main(args.ckpt, args.n, args.lang)
