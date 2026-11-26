# fill-the-gap

**Autoregressive vs. masked-diffusion code models, trained from scratch on a 5M-token Python budget.**

🚧 Work in progress. Results coming.

## Question
On a small, repeated code corpus, which training objective (next-token, fill-in-the-middle,
or masked diffusion) generalizes better as epochs grow, which fills masked code spans more
correctly, and can the diffusion model's own denoising entropy flag wrong infills?

## Status
- [x] Data pipeline: repo-level split, dedup, 8k byte-level BPE, 5M-token train budget
- [x] One transformer (causal / bidirectional flag) + AR, AR-FIM, MDLM objectives, unit tests
- [ ] Training runs (1 → 128 epochs)
- [ ] Infill evaluation
- [ ] Entropy-spike analysis
- [ ] Demo

## Reproduce (so far)
    pip install -r requirements.txt
    python data/prepare.py --near-dedup
    python tests/test_phase2.py

See [NOTES.md](NOTES.md) for hypotheses and design decisions.