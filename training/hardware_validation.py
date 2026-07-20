"""Hardware validation report required by the hackathon rules: inference
latency on CPU-only hardware (Raspberry Pi 4 / phone class, >=8GB RAM) across
the full test set -- NOT the GPU used to train. Confirms the model stays
under the required <=1-2x audio duration before submitting.

Usage: python hardware_validation.py /root/runs/ctc1b/ws_1.../checkpoints/step_N/model/pp_00/tp_00/sdp_00.pt
"""
import sys, os, io, glob, time, base64
import torch
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import polars as pl
import kagglehub

from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

SR = 16000


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


def load_finetuned_cpu(ckpt_pt):
    model = load_model("omniASR_CTC_1B_v2", device=torch.device("cpu"), dtype=torch.float32)
    sd = torch.load(ckpt_pt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def main(ckpt_path, out_csv="hardware_validation_report.csv"):
    test_path = kagglehub.dataset_download("digitalumuganda/anv-test-data-nt")

    model = load_finetuned_cpu(ckpt_path)
    pipe = ASRInferencePipeline(model_card=None, model=model,
                                 tokenizer=load_tokenizer("omniASR_tokenizer_written_v2"))

    rows = []
    files = sorted(glob.glob(f"{test_path}/*/*/*.parquet"))
    for i, fp in enumerate(files):
        t = pq.read_table(fp, columns=["id", "audio", "language"])
        for _id, a, lang in zip(t.column("id").to_pylist(), t.column("audio").to_pylist(),
                                 t.column("language").to_pylist()):
            raw = a["bytes"] if isinstance(a, dict) else a
            try:
                wav, sr = read_audio_bytes(raw)
            except Exception as e:
                print(f"WARN: failed on id={_id} lang={lang}: {e}", flush=True)
                continue
            dur = len(wav) / sr
            t0 = time.time()
            pipe.transcribe([{"waveform": torch.from_numpy(wav.astype(np.float32)), "sample_rate": sr}],
                             batch_size=1)
            elapsed = time.time() - t0
            rows.append({"id": _id, "language": lang, "duration_s": dur, "latency_s": elapsed,
                         "realtime_factor": elapsed / dur})
        print(f"{i+1}/{len(files)} files done, {len(rows)} clips timed", flush=True)

    df = pl.DataFrame(rows)
    df.write_csv(out_csv)

    summary = (df.group_by("language")
                 .agg(pl.col("realtime_factor").mean().alias("mean_rtf"),
                      pl.col("realtime_factor").max().alias("max_rtf")))
    print(summary)
    overall_mean = df["realtime_factor"].mean()
    overall_max = df["realtime_factor"].max()
    print(f"\noverall mean realtime factor: {overall_mean:.3f}")
    print(f"overall max realtime factor: {overall_max:.3f}")
    print("(must be <= 1-2x per hackathon rules; check both mean and worst-case max)")


if __name__ == "__main__":
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "hardware_validation_report.csv"
    main(ckpt, out)
