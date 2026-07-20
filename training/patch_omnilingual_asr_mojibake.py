"""Patches the cloned omnilingual-asr repo's text pipeline in two ways:

1. Applies the mojibake fix (see generate_submission.py/evaluate.py's
   clean_pred docstring for the full character-level explanation) to
   TRAINING TEXT itself, at data-load time, before tokenization -- rather
   than only at inference time on model output.
2. Hardens the empty-text filter against null (NaN/float) transcription
   cells, which otherwise crash the data pipeline with a TypeError.

This is a source patch, not a monkeypatch: fairseq2 recipe internals import
`load_tokenizer`/pipeline helpers via `from x import y`, which binds a local
reference at import time and would silently ignore an external monkeypatch of
the origin module. Editing the actual pipeline-construction call site in the
cloned repo is the reliable way to guarantee the hook runs.

Why this location, and why it's a BETTER fix than the inference-side one:
the raw stored `text` field still has the exact 2-character mojibake sequence
(e.g. "Ä©" for a corrupted "ĩ") intact at this point -- unlike inference
output, where the tokenizer has already collapsed both the upper- and
lower-case corrupted forms to the same single <unk> token, forcing a
case-guessing heuristic. Here the fix is exact, no guessing needed.

Run once against a fresh `git clone https://github.com/facebookresearch/omnilingual-asr`,
before installing/running training. Idempotent: safe to re-run against an
already-patched checkout (it's a no-op update, not a duplicate insert, if the
markers are already present).

Usage: python patch_omnilingual_asr_mojibake.py [path/to/omnilingual-asr]
"""
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "omnilingual-asr"
TEXT_PY = f"{REPO}/src/omnilingual_asr/datasets/utils/text.py"
ASR_TASK_PY = f"{REPO}/src/omnilingual_asr/datasets/tasks/asr_task.py"

FIX_FUNC = '''

# AfriVoices data-prep bug: some source text had UTF-8 diacritics (ĩ/ũ/Ĩ/Ũ)
# mis-decoded upstream as Latin-1, producing two-char mojibake the tokenizer
# cannot represent (it emits <unk> there instead, and the model learns to
# reproduce that at inference). Reversed deterministically here -- the raw
# byte sequence unambiguously identifies upper vs. lower case, unlike trying
# to recover it after the tokenizer has already collapsed both cases to the
# same <unk> id (the problem the inference-side post-processing fix has to
# work around).
_MOJIBAKE_MAP = {
    "\\u00c5\\u00a9": "\\u0169",  # Å© -> ũ
    "\\u00c4\\u00a9": "\\u0129",  # Ä© -> ĩ
    "\\u00c5\\u00a8": "\\u0168",  # Å¨ -> Ũ
    "\\u00c4\\u00a8": "\\u0128",  # Ä¨ -> Ĩ
    "\\u00e2\\u20ac\\u2122": "\\u0027",  # Luo double-encoded right-quote (â€™ -> ')
}
_MOJIBAKE_RE = re.compile("|".join(re.escape(k) for k in _MOJIBAKE_MAP))


def _fix_mojibake_str(t: str) -> str:
    import unicodedata
    # Order matters: NFKC has a *compatibility* decomposition for U+00A8 (the
    # diaeresis used in the Ũ/Ĩ mojibake pairs) into "space + combining
    # diaeresis", which would destroy the exact 2-char sequence below before
    # this ever gets to match it. Substitute the mojibake FIRST, normalize after.
    t = _MOJIBAKE_RE.sub(lambda m: _MOJIBAKE_MAP[m.group(0)], t)
    return unicodedata.normalize("NFKC", t)  # Somali stylized-unicode captions -> plain letters


def fix_mojibake(
    builder: DataPipelineBuilder, text_selector: str
) -> DataPipelineBuilder:
    """Expects the data to have a `text_selector` field holding a raw str."""
    return builder.map(_fix_mojibake_str, selector=text_selector)
'''


def patch_text_py():
    with open(TEXT_PY, "r", encoding="utf-8") as f:
        src = f.read()
    if "fix_mojibake" in src:
        print(f"{TEXT_PY}: already patched, skipping")
    else:
        src = src.replace(
            "from fairseq2.data.tokenizers import TokenEncoder\n",
            "from fairseq2.data.tokenizers import TokenEncoder\nimport re\n",
            1,
        )
        src += FIX_FUNC
        print(f"{TEXT_PY}: patched (mojibake)")

    # Some source rows have a null (NaN, i.e. a float) transcription cell
    # rather than a missing/empty string -- len(nan) raises TypeError instead
    # of just failing the emptiness check (hit again on a fresh HF pull of
    # the dataset, since this was previously only ever patched as a one-off
    # local row-scrub on the original training pod, never pushed back to the
    # dataset itself). Fixing it here at the source keeps it fixed regardless
    # of where the data was pulled from.
    old_filter = (
        "    def is_not_empty(example: Dict[str, Any]) -> bool:\n"
        "        return len(example[text_selector]) > 0\n"
    )
    if old_filter in src:
        new_filter = (
            "    def is_not_empty(example: Dict[str, Any]) -> bool:\n"
            "        text = example[text_selector]\n"
            "        return isinstance(text, str) and len(text) > 0\n"
        )
        src = src.replace(old_filter, new_filter, 1)
        print(f"{TEXT_PY}: patched (null-text filter)")
    elif "isinstance(text, str) and len(text) > 0" in src:
        print(f"{TEXT_PY}: null-text filter already patched, skipping")
    else:
        raise AssertionError("is_not_empty block not found verbatim -- omnilingual-asr source may have changed")

    with open(TEXT_PY, "w", encoding="utf-8") as f:
        f.write(src)


def patch_asr_task_py():
    with open(ASR_TASK_PY, "r", encoding="utf-8") as f:
        src = f.read()
    if "fix_mojibake" in src:
        print(f"{ASR_TASK_PY}: already patched, skipping")
        return

    src = src.replace(
        "from omnilingual_asr.datasets.utils.text import (\n    encode_text,\n    filter_empty_text,",
        "from omnilingual_asr.datasets.utils.text import (\n    encode_text,\n    filter_empty_text,\n    fix_mojibake,",
        1,
    )

    old_block = (
        "        builder = encode_text(\n"
        "            builder,\n"
        "            text_encoder=tokenizer.create_encoder(),\n"
        "            text_selector=text_selector,\n"
        "        )\n"
    )
    assert old_block in src, "encode_text block not found verbatim -- omnilingual-asr source may have changed"
    new_block = "        builder = fix_mojibake(builder, text_selector=text_selector)\n\n" + old_block
    src = src.replace(old_block, new_block, 1)

    with open(ASR_TASK_PY, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"{ASR_TASK_PY}: patched")


if __name__ == "__main__":
    patch_text_py()
    patch_asr_task_py()
