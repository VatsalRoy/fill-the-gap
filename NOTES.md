# NOTES — fill-the-gap

## Problem statement

Companies often have a small private codebase and need to train or adapt a model on it.
With little data, the same tokens must be repeated for many epochs, and models start to
memorize instead of generalize. I train one transformer three ways (next-token, next-token
with fill-in-the-middle, and masked diffusion) on the same fixed 5M tokens of Python, and
measure (1) which objective keeps generalizing longer as epochs grow, (2) which fills a
missing span of code most correctly, and (3) whether the diffusion model's own uncertainty
during decoding can flag wrong infills.

## Papers

### Homebrewed SLMs — Brom, arXiv 2610.40533 (Oct 29, 2026)

- Trains ~11–12M-parameter models from scratch on one consumer GPU, with its own tokenizer.
- Data: Brazilian Portuguese prose and federal laws.
- AR and masked-diffusion generation share one network.
- Preregistered protocol: hypotheses and minimum effect fixed before training.
- Floor: a 6-gram Kneser-Ney model. Yardsticks: public 160M and 630M models.
- Key result: AR peaks at 64 epochs then memorizes; masked diffusion keeps improving up to 256.
- At best checkpoints, the diffusion *bound* is 0.06 bits/byte worse than AR's exact
likelihood, which they report as within their preregistered minimum effect.
- **Copying:** the one-backbone design, preregistration, an n-gram floor, epoch sweep.
- **Different:** code instead of prose/law; I add AR-FIM so diffusion doesn't get a free
win on infilling; I report bits per *token* (they use bits per *byte*); I add an
entropy-spike study.

### MDLM — Sahoo et al., NeurIPS 2024 (kuleshov-group/mdlm)

- Masked diffusion = hide a random fraction t of tokens, predict them all at once.
- Training loss: cross-entropy on masked positions only, weighted by 1/t. This makes the
loss a valid upper bound (ELBO) on negative log-likelihood.
- "SUBS" parametrization: the model never predicts ; unmasked tokens are kept as-is.
- Low-discrepancy sampling of t across the batch reduces loss variance.
- Generation: start fully masked, unmask over S steps.
- Their perplexity numbers are bounds, not exact values. I must label mine the same way.

### Rethinking in Spikes / StepRefine — Xu et al., arXiv 2611.12901 (Nov 2026)

- Observation: during diffusion decoding, mask entropy generally falls step by step, but a
few steps show localized entropy *spikes* that are closely linked to hallucinations.
- Detection: a step is anomalous when its entropy deviates from its neighbouring window.
- Their explanation: committing a discrete token throws away the probability mass on the
other candidates, which hurts exactly at uncertain steps.
- StepRefine: at spike steps, switch to continuous refinement with probability-weighted
embeddings; a controller returns to normal decoding when stable.
- Studied large multimodal MDLMs, not small code models.
- **Copying:** spike detection rule (ΔH_t > mean + std over previous w steps).
- **Different:** a *discrete* fix (re-mask spike tokens and decode again), not their
continuous refinement — continuous refinement is future work.
- ⚠️ My project brief says "~80% of hallucinated tokens fall in spike regions." This is NOT
in the abstract. Check the paper body before quoting it; if I can't find it, don't cite it.

### FIM — Bavarian et al., 2022, arXiv 2207.14255

- Standard trick to let left-to-right models infill: reorder a document as
  ```
   prefix  suffix  middle and train with normal next-token loss.
  ```
- With FIM rate ~0.5, left-to-right ability is barely hurt ("FIM for free").
- This is how production code completion models (Codex-style) infill.
- **Different:** I apply FIM per 256-token window, not per document.

## Hypotheses (written before training)

- **H1 (repetition):** AR and AR-FIM validation loss stop improving earlier (somewhere in
16–64 epochs) than MDLM. MDLM either keeps improving to 128 epochs or degrades more slowly.
Code is more repetitive than prose, so the AR peak could come even earlier than 64.
- **H2 (infilling):** MDLM beats plain AR (which can't see the suffix) on exact match and
ast.parse validity. MDLM roughly ties AR-FIM overall. AR-FIM may still win on single lines.
- **H3 (spikes):** Tokens committed at entropy-spike steps are wrong more often than tokens
committed at normal steps.
- **H4 (fix):** Re-masking spike-step tokens and decoding them again raises infill exact
match, at the cost of a few extra forward passes.

## Minimum meaningful effect (fixed before training)

- Bits/token: a difference < 0.05 counts as "no real difference."
(MDLM is an upper bound, so a small MDLM deficit may not be a real deficit.)
- Infill exact match: a difference < 3 percentage points counts as "no real difference."
- H3: spike-step error rate must be at least 1.5× the normal-step error rate.
- H4: the fix must gain ≥ 2 EM points to be worth its extra cost.

## Dataset (from data/stats.json)

- Source: codeparrot/codeparrot-clean-valid — 61,373 Python files.
- After .py + <50 KB filter: 59,656. After exact dedup: 59,655.
- After MinHash near-dedup (threshold 0.85, 5-word shingles): 58,994 (661 removed).
Small because the "clean" dataset was already deduplicated by its authors.
- Split by repository hash (90/5/5), no repo in two splits (asserted in code):
  - train pool: 53,172 files / 35,970 repos
  - val: 2,963 files / 2,038 repos — 6.2M tokens
  - test: 2,859 files / 1,920 repos
- Train budget: 4,999,936 tokens (19,531 windows × 256). Fresh-data control: same size.
- Tokenizer: byte-level BPE, 8,000 vocab, trained on train split only.
Special ids: PAD 0, EOS 1, MASK 2, PRE 3, SUF 4, MID 5.

## Decisions log

- Model: pre-norm transformer, RMSNorm, RoPE, SwiGLU, tied embeddings.
d_model 384, 6 layers, 6 heads → 10.6M non-embedding params (13.7M total).
- One flag switches attention: causal (AR, AR-FIM) vs bidirectional (MDLM).
- MDLM: no time conditioning (MDLM works without it).
- FIM: rate 0.5, PSM format, applied per window.
- Learning rate: AR / AR-FIM 3e-3 and MDLM 1e-3 in the tiny overfit test. MDLM would not
overfit at 3e-3 — its loss is noisier because each step hides a different random share of
tokens. Gradient clipping 1.0 for all. (Full-run lr to be set in Phase 3.)
- Validation during training: fixed 2,000-window slice; full 6.2M-token val at the end.
- I report bits per token. MDLM's number is always labelled as an upper bound.
- Compute matched by tokens seen, not FLOPs.
- Infill set: 600 items (200 line / 200 block / 200 expr) from held-out test repos,
  ≤3 items per repo, one item per function, functions ≤240 tokens.
  Comment, docstring, blank and def lines never hidden. Split in token space
  (whole function tokenized once) to avoid seam tokens. MDLM gets len(middle_ids) masks.

## Open questions / to check

- [ ] Find the "80% of hallucinated tokens" claim in the Spikes paper body, or drop it.
- [ ] Check that no file in train contains literal "", "", etc. (count of ids 0, 2–5 in train.npy should be 0).
- [ ] Decide checkpoint-selection rule before looking at test results (use val loss only).

## Results — training curves (seed 0, val = 1,000 windows; MDLM = 4-level bound)
| Model  | Best val bits/token | at epoch | val @ epoch 128 | train−val gap @ 128 |
|--------|--------------------:|---------:|----------------:|--------------------:|
| AR     | 4.12                | 4        | 8.25            | 7.8                 |
| AR-FIM | 4.11                | 4        | 7.62            | 7.2                 |
| MDLM   | 4.54 (upper bound)  | 128      | 4.54            | 1.1                 |

- H1 direction confirmed, timing not: AR peaked at epoch 4 (predicted 16–64; paper: 64 on prose).
  MDLM improved monotonically to 128 and was still improving.
- Best-vs-best, AR-FIM beats the MDLM bound by 0.43 bits/token (> 0.05 minimum effect).
  The bound is loose, but unlikely by 0.4 bits. Robust ≠ better on likelihood.
- FIM was free: AR-FIM ≈ AR at the best checkpoint, slightly less overfitting.
- Likely causes of early AR peak: 5M tokens, repetitive code, 14M params, no dropout,
  constant lr. Ablation planned: AR + dropout 0.1.
- Eval checkpoints are coarse (2, 4, 8); true AR minimum is somewhere in 2–8.