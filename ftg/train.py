"""
One training script for all three objectives.

  python -m ftg.train --objective ar     --out runs/ar
  python -m ftg.train --objective ar_fim --out runs/ar_fim
  python -m ftg.train --objective mdlm   --out runs/mdlm

Fair-comparison rules built in:
  * same model size, same data, same data ORDER (seeded per epoch), same lr schedule
  * compute matched by tokens seen (epochs), not FLOPs
  * constant lr after warmup, so a checkpoint at epoch 16 really is "a model trained
    for 16 epochs", not "a model half-way through a cosine schedule"
"""
import argparse, csv, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from ftg.model import Config, Transformer
from ftg.objectives import LOSSES, MASK, mdlm_logits

LN2 = math.log(2)
SAVE_EPOCHS = [1, 4, 16, 32, 64, 128]
EVAL_EPOCHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128]


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def eval_bits_per_token(model, data, objective, device, batch=64, mdlm_k=4, amp=True):
    """
    AR / AR-FIM: exact next-token NLL on plain left-to-right text, in bits/token.
    MDLM: ELBO UPPER BOUND, averaged over mdlm_k noise levels per row on a fixed,
          stratified t grid with fixed mask seeds -> same number every time for the
          same weights, so checkpoints are comparable.
    """
    model.eval()
    g = torch.Generator(device=device).manual_seed(1234)
    total, n = 0.0, 0
    for i in range(0, len(data), batch):
        x = torch.from_numpy(data[i:i + batch].astype(np.int64)).to(device)
        B, T = x.shape
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
            if objective == "mdlm":
                row = torch.zeros(B, device=device)
                for k in range(mdlm_k):
                    t = torch.full((B,), (k + 0.5) / mdlm_k, device=device)
                    m = torch.rand(B, T, device=device, generator=g) < t[:, None]
                    xt = torch.where(m, torch.full_like(x, MASK), x)
                    ce = F.cross_entropy(mdlm_logits(model, xt).float().transpose(1, 2), x,
                                         reduction="none")
                    row += (ce * m).sum(1) / t / T
                total += (row / mdlm_k).sum().item()
            else:
                logits = model(x[:, :-1]).float()
                ce = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
                total += ce.mean(1).sum().item()
        n += B
    model.train()
    return total / n / LN2


# ------------------------------------------------------------------ helpers
def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if args.schedule == "cosine":
        p = (step - args.warmup) / max(1, args.total_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    return args.lr  # constant


def save(path, model, cfg, extra, opt=None, scaler=None):
    state = {"model": model.state_dict(), "config": cfg.__dict__, **extra}
    if opt is not None:
        state["opt"], state["scaler"] = opt.state_dict(), scaler.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)            # atomic: a crash never leaves a half-written file


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=list(LOSSES))
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-file", default="train.npy", help="use control.npy for the fresh-data run")
    p.add_argument("--epochs", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--eval-windows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--max-hours", type=float, default=11.5, help="stop safely before Kaggle's 12 h limit")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not args.no_amp      # fp16 on T4/P100 (no fast bf16 there)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    train = np.load(os.path.join(args.data_dir, args.train_file))
    val = np.load(os.path.join(args.data_dir, "val.npy"))[: args.eval_windows]
    train_eval = train[: args.eval_windows]          # same size slice -> train/val gap
    steps_per_epoch = len(train) // args.batch
    args.total_steps = steps_per_epoch * args.epochs

    cfg = Config(seq_len=train.shape[1], d_model=args.d_model, n_layers=args.n_layers,
                 n_heads=args.n_heads, causal=(args.objective != "mdlm"))
    model = Transformer(cfg).to(device)
    decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
    no_decay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"))
    scaler = torch.amp.GradScaler(enabled=amp)
    fwd = torch.compile(model) if args.compile else model

    # ---- resume if a previous session died
    start_epoch, step, best = 0, 0, float("inf")
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best = ck["epoch"], ck["step"], ck["best"]
        print(f"resumed from epoch {start_epoch}, step {step}")

    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args) | {"params_non_emb": model.num_params()}, f, indent=2)
    log_path = os.path.join(args.out, "log.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["epoch", "step", "tokens_seen", "lr", "train_loss_running",
                    "train_bpt", "val_bpt", "minutes"])

    print(f"{args.objective}: {model.num_params()/1e6:.1f}M non-emb params | "
          f"{steps_per_epoch} steps/epoch | {args.total_steps} total | device {device}")
    t0, run, run_n = time.time(), 0.0, 0
    tok_per_step = args.batch * train.shape[1]

    for epoch in range(start_epoch, args.epochs):
        # Same order for every objective: the permutation depends only on seed + epoch.
        order = np.random.default_rng(args.seed * 1000 + epoch).permutation(len(train))
        for b in range(steps_per_epoch):
            idx = np.sort(order[b * args.batch:(b + 1) * args.batch])
            x = torch.from_numpy(train[idx].astype(np.int64)).to(device, non_blocking=True)
            for g_ in opt.param_groups:
                g_["lr"] = lr_at(step, args)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                loss = LOSSES[args.objective](fwd, x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += loss.item(); run_n += 1; step += 1

            if step == 50:   # early speed report -> tells you the run's real length
                dt = time.time() - t0
                eta = dt / 50 * (args.total_steps - step) / 3600
                print(f"speed: {50 * tok_per_step / dt:,.0f} tok/s | ETA for full run ≈ {eta:.1f} h")
            if args.max_steps and step >= args.max_steps:
                break
        ep = epoch + 1
        minutes = (time.time() - t0) / 60
        extra = {"epoch": ep, "step": step, "best": best}

        if ep in EVAL_EPOCHS or ep == args.epochs or args.max_steps:
            vb = eval_bits_per_token(model, val, args.objective, device, amp=amp)
            tb = eval_bits_per_token(model, train_eval, args.objective, device, amp=amp)
            w.writerow([ep, step, step * tok_per_step, f"{lr_at(step, args):.2e}",
                        f"{run / max(run_n, 1) / LN2:.4f}", f"{tb:.4f}", f"{vb:.4f}", f"{minutes:.1f}"])
            log.flush()
            tag = " (upper bound)" if args.objective == "mdlm" else ""
            print(f"epoch {ep:3d} | train {tb:.3f} | val {vb:.3f} bits/token{tag} | {minutes:.0f} min")
            run, run_n = 0.0, 0
            if vb < best:
                best = extra["best"] = vb
                save(os.path.join(args.out, "best.pt"), model, cfg, extra | {"val_bpt": vb})
        if ep in SAVE_EPOCHS:
            save(os.path.join(args.out, f"epoch{ep:03d}.pt"), model, cfg, extra)
        save(last, model, cfg, extra, opt, scaler)       # resume point, every epoch

        if args.max_steps and step >= args.max_steps:
            break
        if (time.time() - t0) / 3600 > args.max_hours:
            print("time limit reached, saved last.pt -- rerun the same command to resume")
            break
    log.close()
    print("done")


if __name__ == "__main__":
    main()