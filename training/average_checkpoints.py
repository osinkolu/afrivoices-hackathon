"""Averages the weights of several late-training checkpoints into a single
new checkpoint (a standard technique, e.g. used in the original Transformer
paper's last-N-checkpoint averaging) -- NOT ensembling. The output is one
state_dict with the same architecture and parameter count as any other
checkpoint; at inference time it's indistinguishable from a normally-trained
single model (same one forward pass, same latency/RAM profile). Works best
on checkpoints from the same converged/decayed region of training, which is
exactly what step_22000/23000/24000 are (all near the fully-annealed end of
the v3b run's LR schedule).

Usage: python average_checkpoints.py OUT.pt CKPT1.pt CKPT2.pt [CKPT3.pt ...]
"""
import sys
import torch


def average_state_dicts(paths):
    state_dicts = [torch.load(p, map_location="cpu") for p in paths]
    avg = {}
    for key in state_dicts[0]:
        tensors = [sd[key] for sd in state_dicts]
        if tensors[0].is_floating_point():
            avg[key] = sum(t.float() for t in tensors) / len(tensors)
            avg[key] = avg[key].to(tensors[0].dtype)
        else:
            # non-float buffers (e.g. int counters) aren't meaningful to
            # average -- keep the latest checkpoint's value.
            avg[key] = tensors[-1]
    return avg


def main(out_path, ckpt_paths):
    print(f"averaging {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"  {p}")
    avg = average_state_dicts(ckpt_paths)
    torch.save(avg, out_path)
    print(f"wrote averaged checkpoint to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
