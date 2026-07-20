# I didn't write a single line of code for this — here's how a solo, AI-directed entry landed 10th on AfriVoices EAC

## Background

I'm Olufemi Victor, a student at Carnegie Mellon University Africa in Kigali, Rwanda — originally from Nigeria. This was a solo entry, my first time entering a speech competition, and I set myself a constraint on top of the competition's own rules: build the entire thing — data pipeline, training infrastructure across three different cloud providers, debugging, evaluation, error analysis — through natural-language direction to an AI coding agent (Claude Code), and never write or edit a line of code by hand myself.

I want to be upfront about why that framing matters here and why it doesn't: it's not the point of this post, it's the *lens*. The actual work — finding and fixing a root-cause data corruption bug, tuning language balance, chasing down why WER plateaued where it did — is real engineering, and I think it stands on its own regardless of who typed it. What's genuinely interesting is what that process looked like in practice: where it was rock-solid, where I had to catch its mistakes, and what a multi-day, multi-cloud, crash-and-recover ML project looks like when directed rather than hand-coded.

Final result: **0.47017** mean WER, 10th place.

## The setup

- **Model**: `omniASR_CTC_1B_v2` (Meta's Omnilingual ASR family) — a wav2vec2-based CTC encoder, ~1B parameters, character-level tokenizer. Chosen for non-autoregressive, CPU-friendly inference and language-agnostic `transcribe()` — no language-ID step needed, which matters when your test set doesn't reliably ship metadata.
- **Compute**: a single NVIDIA A40 (48GB) on RunPod, bf16, dynamic length-based batching (~1.5M audio samples/step), gradient accumulation over 8 micro-batches. Roughly 65-70 total GPU-hours across every run. Earlier data-pipeline work (pulling and converting six languages' worth of raw source datasets, building a from-scratch speaker-disjoint split) ran separately on a CPU-only VM.
- **Decoding**: plain greedy CTC. No language model, no beam search, no shallow fusion. I'll come back to why that's probably the single biggest thing I'd change with more time.

Full code: [github.com/osinkolu/afrivoices-hackathon](https://github.com/osinkolu/afrivoices-hackathon)
Weights: [huggingface.co/Professor/afrivoices-ctc1b-checkpoints](https://huggingface.co/Professor/afrivoices-ctc1b-checkpoints)

## Finding #1: the mojibake bug (the single biggest lever)

This started from a throwaway observation, not a deliberate audit: I saw someone's unrelated Omnilingual ASR inference code online handling a stray Unicode character (`⁇`, U+2047) in model output, and asked whether that might explain something odd I'd noticed in our own predictions. It did.

A chunk of the Kikuyu (dominant), Luo, and Somali training text had been double-encoded upstream: UTF-8 bytes for diacritics like `ĩ`/`ũ`/`Ĩ`/`Ũ` had been misread as Latin-1/CP1252 somewhere in the data pipeline, producing two-character garbage sequences (`Ä©`, `Å©`, `Ä¨`, `Å¨`) that the ASR tokenizer has no vocabulary entry for. The tokenizer's only option is to emit `<unk>` — and a CTC model trained on that target learns to reproduce exactly what it was shown, `<unk>` and all, at inference time.

The fix went through two stages:

1. **Inference-side patch** (fast, safe, no retraining): reverse-engineer the corruption pattern and reconstruct the correct character from the model's `<unk>`-laden output. Because both the uppercase and lowercase corrupted forms collapse to the *same* `<unk>` token at that point, case had to be guessed heuristically (default lowercase, capitalize after sentence-final punctuation). This alone took the score from 0.595 → 0.52907.
2. **Root-cause fix** (slower, better): patch the actual training data-loading pipeline (`omnilingual-asr`'s `datasets/utils/text.py`) so the correction happens *before* tokenization, on the raw stored text, where the corruption pattern is unambiguous (no case-guessing needed — `©` vs `¨` deterministically tells you upper vs lower case). Retraining on the corrected pipeline: 0.52907 → 0.49859, confirmed via a held-out speaker-disjoint dev-set comparison (~21% relative WER reduction) before ever touching the leaderboard.

## Finding #2: I made the same mistake the eventual first-place team explicitly warned about

After the competition closed, I read the winning team's excellent writeup (worth reading in full) and noticed they'd caught something I hadn't: *"the provided Kikuyu dev was mojibake-corrupted and would have selected the best corruption-writer"* — meaning they fixed their dev-set **reference** text too, not just the training text.

That made me go check my own numbers. Our root-cause fix patched the *training* data loader, but our `evaluate.py` script reads dev-set reference text directly from the raw parquet files — completely bypassing that patch. So I pulled a fresh copy of the dev data and checked, byte by byte:

```
277/9,962 Kikuyu dev rows (2.8%) still contain the raw, uncorrected mojibake sequence
```

I did not catch this during the competition. It's a real gap, and I want to be honest about its actual impact rather than either hiding it or overstating it: since every one of my own checkpoints was evaluated against the *same* flawed 2.8%, the *relative* comparisons I made every model-selection decision on (which checkpoint beat which) are very likely still valid — the corruption penalizes all candidates equally. But the *absolute* WER numbers I quoted throughout for Kikuyu, including the final 0.592 dev-set figure, are very slightly inflated (pessimistic), since correctly-reconstructed predictions were occasionally being compared against a reference that was still wrong.

Lesson: when you patch a bug, check *every* place the corrupted artifact could still be lurking, not just the one you were looking at. I fixed the training text and declared victory; I never re-checked the evaluation harness sitting right next to it.

## Finding #3: one long training run beats many short ones

I ran training in several segments rather than one continuous shot (warm-starting from a checkpoint each time meant restarting the tri-stage LR schedule — this is a training-recipe constraint of only saving model weights, not optimizer/scheduler state, between checkpoints). The pattern was stark and consistent:

| Approach | Steps | Relative WER change |
|---|---|---|
| Mojibake fix, fresh 2,000-step anneal | 2,000 | **-21%** |
| Same anneal type, tripled to 6,000 steps | 6,000 | -2.3% |
| One long continuous 24,000-step anneal | 24,000 | -6.5% (vs. the 6,000-step checkpoint) |

Going from 2,000 to 6,000 steps within the *same* short-restart pattern gave almost nothing. Switching to one long continuous cycle (warmup → hold → decay, uninterrupted) gave a real, repeatable further gain. My read: every restart's warmup phase partially undoes the low-LR convergence the previous cycle had just achieved. If you're warm-starting repeatedly, minimize how many times you do it, not just how many total steps you accumulate.

## Finding #4: the direction-of-a-formula bug

The mixture sampler weights each language by `weight = (hours/total_hours)^beta`. I wanted to push more training exposure toward our two weakest languages (Kalenjin, Maasai), and my first instinct — confidently stated, in my own working notes — was "increase beta toward 1.0 to upweight low-resource languages."

That's backwards. `beta=1` means fully proportional to natural data volume (which *favors* Swahili, our highest-resource language); `beta→0` means fully equal weighting regardless of volume. I caught this specifically because I'd made a habit of verifying claims against the actual library source before acting on them, rather than trusting my own restated understanding — in this case, reading `mixture_parquet_storage.py`'s actual weight computation directly. Lowering beta from 0.5 to 0.2 (the correct direction) measurably helped Kalenjin and Maasai without hurting anyone else — every one of the 6 languages improved simultaneously at that setting.

## Finding #5: two honest negative results

Not everything I tried worked, and I think that's worth stating plainly rather than only listing what did:

- **Checkpoint averaging** (averaging the weights of the last 3 checkpoints, a standard technique — not ensembling, since it produces a single set of weights with the same inference cost as any other checkpoint): essentially a wash (0.4932 vs. 0.4924 dev WER for the single best checkpoint), likely because checkpoints only 1,000-2,000 steps apart, late in the same decay, are too correlated to give real noise-cancellation benefit.
- **Pushing `beta_language` further, from 0.2 to 0.1**: made things *slightly worse* across the board, including the two languages I was specifically trying to help. Best guess: 0.2 was already close to the useful optimum for this data, and the extra correction started trading away useful Swahili/Kikuyu signal for a benefit that didn't materialize — possibly also because the follow-up training run was too short (3,000 steps) for a fresh cycle to show its full effect.

## Why WER plateaued where it did — a proper look, not a guess

With the leaderboard closed and no test-set references to reverse-engineer against, this section is dev-set analysis, not blind speculation. I inspected actual model output against references directly, rather than trusting aggregate WER alone.

**Annotation-tag artifacts.** Reference transcriptions contain literal transcriber tags like `[cs]` (code-switch boundary) and `[pause]` embedded as text. A correct model should never produce these — they're not spoken words — but WER counts their absence as an error every time. I measured the actual impact by scoring the same predictions against both the raw and tag-stripped reference:

| Language | Tag prevalence in dev | WER inflation from tags |
|---|---|---|
| Maasai | 12.8% of rows | 0.88 percentage points |
| Kalenjin | 10.4% of rows | 0.23 pp |
| Swahili | 4.3% of rows | 0.42 pp |
| Luo | 5.2% of rows | 0.09 pp |
| Kikuyu | 1.4% of rows | 0.02 pp |
| Somali | 1.4% of rows | 0.01 pp |

Real, but modest — nowhere close to explaining the gap to the leaderboard leader on its own.

**Orthographic inconsistency.** Kalenjin and Maasai don't have one standardized spelling convention in this dataset. Looking at real examples, a large share of "errors" are phonetically identical variants — `Walwal nganuk` vs. `Walawal nganok`, `loindab mitait` vs. `laindap mit ait` — b/p, k/g, o/a swaps and word-boundary differences that reflect transcriber convention, not the model mishearing.

**The honest bottom line.** Neither of the above fully explains the remaining gap. The dominant factors are almost certainly genuine training-data scarcity — Kalenjin (86k rows) and Maasai (53k rows) have 6-9x less data than Swahili (503k rows), even after aggressive upweighting — combined with plain greedy CTC decoding's ceiling without a language model to smooth over ambiguity. The first-place team's writeup confirms this last point concretely: their single biggest post-data-fix gain was a per-language KenLM with beam search (greedy 0.39652 → 0.34929). I didn't have time to build that. If I re-ran this competition, that's the first thing I'd add.

## What directing an AI agent was actually like for this

Since that was the constraint I set myself, a genuine assessment, not a sales pitch:

**Where it held up well:** systematic, patient debugging across dozens of distinct failure modes over several days (encoding bugs, OOM tuning, a crashed training run mid-anneal, SSL/network issues, self-referential process-matching bugs); disciplined "verify before deciding" habits — checkpoint decisions were always made against measured dev-set WER, never assumed; catching its own mistakes when explicitly asked to check rather than just report (the `beta_language` direction, the dev-reference-corruption gap above, were both found by going back and checking, not by getting it right the first time).

**Where I had to catch it:** the `beta_language` direction was stated confidently and wrong before I asked for it to be verified against source. A `pkill` command matched its own invocation and silently reported success on a process that was actually still running. A log-monitoring filter for "wandb" once matched the literal value of a wandb API key sitting in an environment-variable dump, and printed it — a real lesson in how a broad keyword filter meant to catch one thing can leak something else entirely.

None of these were fatal, because catching them was part of the process, not an afterthought. But I don't think "you can just let it run" is an honest takeaway — the honest one is that it's a genuine collaboration, and the value was in the verification discipline as much as the execution speed.

## Results, links, and thanks

- **Final score**: 0.47017 mean WER, 10th place
- **Code**: [github.com/osinkolu/afrivoices-hackathon](https://github.com/osinkolu/afrivoices-hackathon)
- **Weights**: [huggingface.co/Professor/afrivoices-ctc1b-checkpoints](https://huggingface.co/Professor/afrivoices-ctc1b-checkpoints) (final checkpoint: `v3b/step_24000`)

Thanks to the organizers (Maseno Center for Applied AI, Maseno University, Digital Umuganda / KenCorpus Consortium) for putting together a competition on six genuinely under-represented languages, and to everyone who shared their approaches after close — reading the first-place team's writeup directly led to finding a real gap in my own evaluation methodology, which is exactly the kind of thing this kind of open post-mortem is for.

Happy to answer questions on any of this.
