"""
The three training objectives. Same model, same data -- only these functions differ.
Special token ids are fixed by data/prepare.py.
"""
import torch
import torch.nn.functional as F

PAD, EOS, MASK, PRE, SUF, MID = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------- 1. Autoregressive
def ar_loss(model, x):
    """Next-token prediction: read tokens 0..T-2, predict tokens 1..T-1. Returns nats/token."""
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))


# ---------------------------------------------------------------- 2. Fill-in-the-middle
def fim_transform(x, rate=0.5, generator=None):
    """
    With probability `rate`, rewrite a row as  <PRE> prefix <SUF> suffix <MID> middle.
    The 3 sentinels need room, so we use the first T-3 tokens of the row as content;
    output length stays T. Rows not chosen are returned unchanged (plain AR).
    Note: applied per 256-token window, not per file (simpler; say so in the README).
    """
    B, T = x.shape
    out = x.clone()
    n = T - 3
    for b in range(B):
        if torch.rand(1, generator=generator).item() >= rate:
            continue
        i, j = sorted(torch.randint(0, n + 1, (2,), generator=generator).tolist())
        c = x[b, :n]
        pre, mid, suf = c[:i], c[i:j], c[j:]
        s = lambda t: torch.tensor([t], dtype=x.dtype, device=x.device)
        out[b] = torch.cat([s(PRE), pre, s(SUF), suf, s(MID), mid])
    return out


def fim_inverse(row):
    """Undo fim_transform on one row (used only in the unit test)."""
    r = row.tolist()
    if r[0] != PRE:
        return r
    si, mi = r.index(SUF), r.index(MID)
    pre, suf, mid = r[1:si], r[si + 1:mi], r[mi + 1:]
    return pre + mid + suf


def ar_fim_loss(model, x, rate=0.5, generator=None):
    return ar_loss(model, fim_transform(x, rate, generator))


# ---------------------------------------------------------------- 3. Masked diffusion
def mdlm_logits(model, xt):
    """MDLM 'SUBS' trick: the model may never predict <MASK> itself."""
    logits = model(xt)
    logits[..., MASK] = float("-inf")
    return logits


def mdlm_loss(model, x, eps=1e-3, t=None):
    """
    1. Pick a noise level t ~ U(eps, 1) per row (t = fraction of tokens hidden).
    2. Hide each token with probability t (replace by <MASK>).
    3. Cross-entropy on the hidden positions only.
    4. Weight by 1/t  (rows with few masks count more -- this makes it a valid ELBO),
       and divide by sequence length -> an upper bound on nats/token.
    """
    B, T = x.shape
    if t is None:
        # Low-discrepancy sampling (from the MDLM repo): evenly spread t across the batch,
        # with a random offset. Same expectation as U(0,1), lower variance.
        u = (torch.rand(1, device=x.device) + torch.arange(B, device=x.device) / B) % 1
        t = eps + (1 - eps) * u
    masked = torch.rand(B, T, device=x.device) < t[:, None]
    xt = torch.where(masked, torch.full_like(x, MASK), x)
    logits = mdlm_logits(model, xt)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1),
                         reduction="none").view(B, T)
    per_row = (ce * masked).sum(1) / t / T
    return per_row.mean()


LOSSES = {"ar": ar_loss, "ar_fim": ar_fim_loss, "mdlm": mdlm_loss}