"""Watches the training run's checkpoints/ directory for new step_N checkpoints
and pushes each (model-only) checkpoint to a private HF repo as a safety net --
run alongside training in a separate shell/session.

RunPod pods can be stopped/terminated; this ensures a dropped session doesn't
cost the run its checkpoints.

Originally this watched a `valid.jsonl` score log to push only the best
checkpoint (matching the upstream tutorial this project started from), but
this fairseq2/omnilingual-asr version doesn't produce that file (metrics land
in `metrics/train.jsonl` instead, and validation triggering turned out to be
harder to pin down than was worth chasing under deadline pressure). Backing up
every checkpoint as it appears is simpler and doesn't depend on validation/
scoring working correctly at all -- the actual best checkpoint gets picked
later via evaluate.py against real dev-set WER.
"""
import os, time, glob

from huggingface_hub import HfApi

OUTPUT_DIR = os.environ.get("CKPT_WATCH_DIR", "/root/runs/ctc1b")
CHECKPOINT_REPO = "Professor/afrivoices-ctc1b-checkpoints"
# Path prefix within CHECKPOINT_REPO -- kept distinct per run so a continued/
# warm-started run's checkpoint numbering (which restarts from step_0, since
# model.path warm-starts don't carry over optimizer/trainer step state) can't
# collide with and silently overwrite an earlier run's same-numbered steps.
REMOTE_PREFIX = os.environ.get("CKPT_REMOTE_PREFIX", "")
POLL_SECONDS = 60

api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True)


def find_checkpoint_steps():
    pattern = os.path.join(OUTPUT_DIR, "ws_1.*", "checkpoints", "step_*",
                            "model", "pp_00", "tp_00", "sdp_00.pt")
    steps = {}
    for p in glob.glob(pattern):
        # fairseq2 writes checkpoints to a "step_N.tmp" dir first, then
        # atomically renames it to "step_N" -- the glob above can catch a
        # checkpoint mid-rename, so the segment after "step_" isn't always
        # a plain integer (crashed on "1000.tmp" once already). Skip those;
        # they'll show up correctly named on a later poll once renamed.
        raw = p.split("step_")[1].split(os.sep)[0]
        if not raw.isdigit():
            continue
        steps[int(raw)] = p
    return steps


def main():
    pushed_steps = set()
    print(f"watching {OUTPUT_DIR} for new checkpoints, polling every {POLL_SECONDS}s", flush=True)

    while True:
        try:
            steps = find_checkpoint_steps()
            for step in sorted(steps):
                if step in pushed_steps:
                    continue
                ckpt = steps[step]
                # skip if still being written (size unstable across a short pause)
                size_a = os.path.getsize(ckpt)
                time.sleep(2)
                size_b = os.path.getsize(ckpt)
                if size_a != size_b:
                    continue

                print(f"new checkpoint step={step} -- uploading {ckpt}", flush=True)
                remote_path = f"{REMOTE_PREFIX}step_{step}/sdp_00.pt"
                api.upload_file(
                    path_or_fileobj=ckpt,
                    path_in_repo=remote_path,
                    repo_id=CHECKPOINT_REPO,
                    repo_type="model",
                )
                # verify: compare uploaded file size against local before trusting the backup
                local_size = os.path.getsize(ckpt)
                info = api.get_paths_info(CHECKPOINT_REPO, [remote_path], repo_type="model")
                remote_size = info[0].size if info else None
                if remote_size == local_size:
                    print(f"verified upload OK step={step} ({local_size} bytes)", flush=True)
                    pushed_steps.add(step)
                else:
                    print(f"WARNING: size mismatch step={step} local={local_size} remote={remote_size}, will retry", flush=True)
        except Exception as e:
            # A backup hiccup (transient network error, an unexpected path
            # shape, etc.) must never take the whole watcher down -- losing
            # backup coverage for the rest of a multi-hour run over one bad
            # poll cycle is a much worse outcome than logging and retrying.
            print(f"WARNING: backup cycle failed, will retry next poll: {e}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
