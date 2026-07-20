import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))

REPOS = [
    "DigitalUmuganda/Afrivoice_Swahili",
    "Anv-ke/kikuyu",
    "Anv-ke/Dholuo",
    "Anv-ke/Somali",
    "Anv-ke/Maasai",
    "Anv-ke/Kalenjin",
    "DigitalUmuganda/Afrivoice",
    "DigitalUmuganda/anv_test_data_nt",
]

for repo in REPOS:
    print(f"\n=== {repo} ===")
    try:
        info = api.dataset_info(repo, files_metadata=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue
    total = 0
    top_dirs = {}
    for s in info.siblings:
        size = s.size or 0
        total += size
        top = s.rfilename.split("/")[0]
        top_dirs[top] = top_dirs.get(top, 0) + size
    print(f"  total files: {len(info.siblings)}  total size: {total/1e9:.2f} GB")
    for d, sz in sorted(top_dirs.items(), key=lambda x: -x[1])[:15]:
        print(f"    {d}: {sz/1e9:.2f} GB")
