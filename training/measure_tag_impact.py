"""Quantifies how much of measured WER is an artifact of transcriber
annotation tags (e.g. "[cs]" for code-switch boundaries, "[pause]") embedded
literally in the reference text -- the model correctly never produces these
(they aren't spoken words), but exact-match WER counts their absence as an
error every time. Scores the same predictions against the raw reference and
against a tag-stripped reference, so the delta isolates this specific
artifact from genuine recognition errors.

Usage: python measure_tag_impact.py CKPT.pt [--n-per-lang N]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import io, re, glob
from collections import defaultdict
import torch
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import jiwer

from fairseq2.data.tokenizers import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from evaluate import (
    clean_pred, prep, load_finetuned, flush_batch, iter_sampled_rows,
    DATA_ROOT, BATCH_ROWS,
)

TAG_RE = re.compile(r"\[[a-zA-Z_ ]+\]", re.IGNORECASE)


def strip_tags(s):
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", s)).strip()


def main(ckpt_path, n_per_lang):
    model = load_finetuned(ckpt_path)
    pipe = ASRInferencePipeline(model_card=None, model=model,
                                 tokenizer=load_tokenizer("omniASR_tokenizer_written_v2"))

    print(f"{'lang':<5} {'n':>5} {'wer_raw':>10} {'wer_notags':>12} {'delta':>8}")
    for lang_dir in sorted(glob.glob(os.path.join(DATA_ROOT, "language=*"))):
        lang = os.path.basename(lang_dir).split("=")[1]
        refs, hyps = [], []
        buffer = []
        for ref_text, raw in iter_sampled_rows(lang_dir, n_per_lang):
            wav, sr = sf.read(io.BytesIO(raw))
            chunks = prep(wav.astype(np.float32))
            buffer.append((ref_text, chunks))
            if len(buffer) >= BATCH_ROWS:
                flush_batch(buffer, pipe, refs, hyps)
                buffer = []
        if buffer:
            flush_batch(buffer, pipe, refs, hyps)

        wer_raw = jiwer.wer(refs, hyps)
        refs_notags = [strip_tags(r) for r in refs]
        wer_notags = jiwer.wer(refs_notags, hyps)
        delta = wer_raw - wer_notags
        print(f"{lang:<5} {len(refs):>5} {wer_raw:>10.4f} {wer_notags:>12.4f} {delta:>8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n-per-lang", type=int, default=300)
    args = ap.parse_args()
    main(args.ckpt, args.n_per_lang)
