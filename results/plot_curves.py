"""
Plot 1: validation bits/token vs epochs (log x), minimum of each curve marked.
Plot 2: train-val gap vs epochs (the memorization picture).

    python results/plot_curves.py      -> results/plot1_val_vs_epochs.png, results/plot2_gap.png

Optional: pass --floor 4.9 to draw the 5-gram floor once it exists (Phase 4).
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Fixed order, colorblind-checked: blue, orange, aqua. Direct labels + markers carry
# identity too, so the chart never relies on color alone.
RUNS = [("ar", "AR", "#257de0", "o"),
        ("ar_fim", "AR-FIM", "#f47230", "s"),
        ("mdlm", "MDLM (upper bound)", "#00b894", "^")]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def load(name):
    with open(os.path.join(HERE, f"{name}_log.csv")) as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    return ep, [float(r["val_bpt"]) for r in rows], [float(r["train_bpt"]) for r in rows]


def style(ax, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("epochs over the same 5M tokens (log scale)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--floor", type=float, help="5-gram floor in bits/token")
    args = p.parse_args()
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    # ---------------- Plot 1
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for name, label, color, marker in RUNS:
        ep, val, _ = load(name)
        ax.plot(ep, val, color=color, lw=2, marker=marker, ms=6, label=label,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        i = min(range(len(val)), key=val.__getitem__)
        ax.scatter([ep[i]], [val[i]], s=160, facecolor="none", edgecolor=INK, lw=1.5, zorder=4)
        offset = {"ar": (30, -24), "ar_fim": (-30, -24), "mdlm": (-10, -28)}[name]
        ax.annotate(f"{label.split(' ')[0]} best {val[i]:.2f} @ epoch {ep[i]}", (ep[i], val[i]),
                    xytext=offset, textcoords="offset points", ha="left" if name == "ar" else "right",
                    va="center", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=7))
        ax.annotate(label, (ep[-1], val[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    if args.floor:
        ax.axhline(args.floor, color=MUTED, ls="--", lw=1.2)
        ax.annotate(f"5-gram floor {args.floor:.2f}", (1, args.floor), xytext=(2, 4),
                    textcoords="offset points", fontsize=9, color=MUTED)
    style(ax, "validation bits per token (lower is better)")
    ax.set_xlim(0.85, 400)
    ax.set_ylim(3.6, 8.6)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.26, 1.0), labelcolor=INK)
    ax.set_title("AR models peak at epoch 4 and then memorize; diffusion keeps improving",
                 loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plot1_val_vs_epochs.png"))

    # ---------------- Plot 2
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for name, label, color, marker in RUNS:
        ep, val, tr = load(name)
        gap = [v - t for v, t in zip(val, tr)]
        ax.plot(ep, gap, color=color, lw=2, marker=marker, ms=6, label=label,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{label}  {gap[-1]:.1f}", (ep[-1], gap[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10, color=INK)
    style(ax, "val − train bits per token (memorization gap)")
    ax.set_xlim(0.85, 400)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("Train-val gap: AR models memorize the training set, MDLM barely does",
                 loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plot2_gap.png"))
    print("saved plot1_val_vs_epochs.png and plot2_gap.png")


if __name__ == "__main__":
    main()