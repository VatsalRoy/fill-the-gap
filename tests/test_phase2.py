"""
Phase 2 checks. Run:  python -m pytest tests/ -s     (or: python tests/test_phase2.py)
Uses a TINY model so it runs on a laptop CPU in about a minute.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ftg.model import Config, Transformer
from ftg.objectives import fim_transform, fim_inverse, ar_loss, ar_fim_loss, mdlm_loss

TINY = dict(vocab_size=8000, seq_len=64, d_model=128, n_layers=2, n_heads=4)


def test_fim_roundtrip():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(6, 8000, (200, 64))
    y = fim_transform(x, rate=1.0, generator=g)
    assert y.shape == x.shape
    for a, b in zip(x, y):
        assert fim_inverse(b) == a[:61].tolist()      # content = first T-3 tokens
    print("FIM round-trip ok")


def test_causal_flag():
    x = torch.randint(6, 8000, (1, 64))
    x2 = x.clone(); x2[0, -1] = 7                      # change only the LAST token
    for causal in (True, False):
        torch.manual_seed(0)
        m = Transformer(Config(**TINY, causal=causal)).eval()
        with torch.no_grad():
            diff = (m(x)[0, :-1] - m(x2)[0, :-1]).abs().max().item()
        if causal:
            assert diff < 1e-5, "causal model peeked at the future!"
        else:
            assert diff > 1e-4, "bidirectional model ignores the future!"
    print("causal / bidirectional flag ok")


def overfit(loss_fn, causal, steps, lr):
    torch.manual_seed(0)
    m = Transformer(Config(**TINY, causal=causal))
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    x = torch.randint(6, 8000, (4, 64))                # one fixed batch
    for s in range(steps):
        loss = loss_fn(m, x)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        if loss_fn is mdlm_loss:
            # MDLM's training loss is random (new t and new masks each step), so judge it
            # at a FIXED mask rate: hide 50% of tokens, can it fill them back in?
            return mdlm_loss(m, x, t=torch.full((4,), 0.5)).item()
        return loss_fn(m, x).item()


def test_overfit_one_batch():
    # Random tokens start at ln(8000) = 9.0 nats. Memorizing should push this near 0.
    # MDLM needs a gentler lr and more steps: its loss is much noisier.
    for name, fn, causal, steps, lr in [("ar", ar_loss, True, 300, 3e-3),
                                        ("ar_fim", ar_fim_loss, True, 300, 3e-3),
                                        ("mdlm", mdlm_loss, False, 1500, 1e-3)]:
        final = overfit(fn, causal, steps, lr)
        print(f"{name:7s} final loss {final:.3f}")
        assert final < 0.5, f"{name} did not overfit one batch"


if __name__ == "__main__":
    test_fim_roundtrip(); test_causal_flag(); test_overfit_one_batch()