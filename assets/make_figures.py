"""Generates the README figures in assets/: results chart (Table 1) and the pipeline GIF."""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e4e3df",
                  ramp4=["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
                  seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
                  accent="#eb6834", track="#f0efec"),
    "dark": dict(bg="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#8f8e86", grid="#383835",
                 ramp4=["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"],
                 seq=["#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
                 accent="#d95926", track="#2a2a28"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

# ---------------- Results chart (Table 1) ----------------
SETTINGS = ["Direct", "Direct + PRM Sel.", "Dynamic", "Dynamic + PRM Sel."]
MODELS = ["Llama-3.1-8B", "Qwen-2.5-7B"]
DATA = {
    "MATH-500": [[43.8, 44.6, 47.2, 65.4], [71.2, 72.0, 74.2, 81.4]],
    "AIME24": [[3.3, 6.67, 6.67, 10.0], [6.67, 6.67, 10.0, 13.3]],
    "AMO-Bench": [[2.0, 2.0, 4.0, 4.0], [2.0, 2.0, 4.0, 4.0]],
}
YMAX = {"MATH-500": 100, "AIME24": 16, "AMO-Bench": 5}


def results_chart(mode):
    t = THEMES[mode]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    w, gap = 0.18, 0.02
    for ax, (name, vals) in zip(axes, DATA.items()):
        ax.set_facecolor(t["bg"])
        ymax = YMAX[name]
        for gi, mvals in enumerate(vals):
            for si, v in enumerate(mvals):
                x = gi + (si - 1.5) * (w + gap) - w / 2
                ax.bar(x + w / 2, v, width=w, color=t["ramp4"][si], edgecolor=t["bg"], linewidth=0.8, zorder=3)
                if si in (0, 3):
                    ax.text(x + w / 2, v + ymax * 0.015, f"{v:g}", ha="center", va="bottom",
                            fontsize=9.5, color=t["ink"] if si == 3 else t["ink2"],
                            fontweight="bold" if si == 3 else "normal", zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(MODELS, color=t["ink2"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, ymax)
        ax.set_title(name, color=t["ink"], fontsize=13, fontweight="bold", loc="left", pad=10)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
        ax.tick_params(colors=t["muted"], length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(t["grid"])
    axes[0].set_ylabel("Accuracy (%)", color=t["ink2"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in t["ramp4"]]
    leg = fig.legend(handles, SETTINGS, loc="upper center", ncol=4, frameon=False,
                     bbox_to_anchor=(0.5, 1.02), fontsize=11, handlelength=1.2)
    for tx in leg.get_texts():
        tx.set_color(t["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / f"results_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


# ---------------- Pipeline GIF (illustrative) ----------------
ROWS = [  # tools, strategy, step scores
    ("CoT + NV", "Lookahead d=3", [0.88, 0.80, 0.85, 0.79, 0.83]),
    ("SR", "Best-of-N N=4", [0.81, 0.62, 0.55, 0.70, 0.58]),
    ("R + CoT", "Beam k=4", [0.90, 0.74, 0.83, 0.66]),
    ("CoT", "Best-of-N N=3", [0.70, 0.52, 0.41, 0.48, 0.55, 0.45]),
    ("CoT + NV + V", "Lookahead d=2", [0.86, 0.90, 0.78, 0.84]),
    ("SR + NV", "Beam k=5", [0.64, 0.71, 0.58, 0.69, 0.60]),
    ("CoT + NV", "Lookahead d=4", [0.92, 0.88, 0.95, 0.91, 0.97]),
    ("CoT + NV", "Best-of-N N=5", [0.55, 0.47, 0.62, 0.38]),
    ("R + SR", "Beam k=3", [0.77, 0.69, 0.81, 0.73, 0.68]),
    ("CoT + S", "Lookahead d=3", [0.60, 0.72, 0.66, 0.57, 0.63]),
]
K = len(ROWS)
MEANS = [float(np.mean(s)) for _, _, s in ROWS]
BEST = int(np.argmax(MEANS))


def draw_frame(mode, n_rows, partial_steps, show_select):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9, 5.6), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.text(3, 58.5, "Adaptive test-time compute: iterate, verify, select", fontsize=15,
            fontweight="bold", color=t["ink"], va="center")
    ax.text(3, 54.6, "Each iteration picks its own tools and compute strategy; the PRM scores every step.",
            fontsize=10.5, color=t["ink2"], va="center")
    hdr_y = 50.5
    for x, lab in [(3, "Iter"), (10, "Tools (A_T)"), (27, "Strategy (A_C)"), (47, "Step scores"),
                   (74, "R(τ) = mean v_t")]:
        ax.text(x, hdr_y, lab, fontsize=9.5, color=t["muted"], va="center")
    ax.text(58.5, hdr_y, "low", fontsize=8.5, color=t["muted"], va="center", ha="right")
    for j, c in enumerate(t["seq"]):
        ax.add_patch(plt.Circle((59.8 + j * 1.25, hdr_y), 0.55, facecolor=c, edgecolor="none"))
    ax.text(68.4, hdr_y, "high", fontsize=8.5, color=t["muted"], va="center")

    def color_for(v):
        idx = int(np.clip((v - 0.35) / 0.65 * (len(t["seq"]) - 1), 0, len(t["seq"]) - 1))
        return t["seq"][idx]

    for i in range(n_rows):
        tools, strat, scores = ROWS[i]
        y = 46 - i * 4.3
        is_best = show_select and i == BEST
        if is_best:
            ax.add_patch(FancyBboxPatch((1.5, y - 1.9), 96.5, 3.8, boxstyle="round,pad=0,rounding_size=1.2",
                                        facecolor="none", edgecolor=t["accent"], linewidth=2.2))
        fade = 1.0 if (not show_select or is_best) else 0.35
        ax.text(3, y, f"I{i+1}", fontsize=10.5, color=t["ink"], va="center", alpha=fade,
                fontweight="bold" if is_best else "normal")
        ax.text(10, y, tools, fontsize=10, color=t["ink2"], va="center", alpha=fade)
        ax.text(27, y, strat, fontsize=10, color=t["ink2"], va="center", alpha=fade)
        k = len(scores) if i < n_rows - 1 else min(partial_steps, len(scores))
        for j in range(k):
            ax.add_patch(plt.Circle((48 + j * 4.0, y), 1.25, facecolor=color_for(scores[j]),
                                    edgecolor=t["bg"], linewidth=1.5, alpha=fade))
        if k == len(scores):
            m = MEANS[i]
            ax.add_patch(FancyBboxPatch((74, y - 1.0), 16, 2.0, boxstyle="round,pad=0,rounding_size=0.8",
                                        facecolor=t["track"], edgecolor="none", alpha=fade))
            ax.add_patch(FancyBboxPatch((74, y - 1.0), 16 * m, 2.0, boxstyle="round,pad=0,rounding_size=0.8",
                                        facecolor=t["accent"] if is_best else t["ramp4"][2],
                                        edgecolor="none", alpha=fade))
            ax.text(91.5, y, f"{m:.2f}", fontsize=10, color=t["ink"], va="center", alpha=fade,
                    fontweight="bold" if is_best else "normal")
    if show_select:
        ax.text(50, 2.2, f"Final answer ŷ taken from iteration I{BEST+1} (highest mean PRM reward)",
                fontsize=11.5, color=t["ink"], ha="center", va="center", fontweight="bold")
    else:
        ax.text(50, 2.2, "Illustrative example, K = 10 iterations", fontsize=9.5, color=t["muted"],
                ha="center", va="center")
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img


def pipeline_gif(mode):
    frames, durations = [], []
    for i in range(1, K + 1):
        n = len(ROWS[i - 1][2])
        for s in (0, n // 2, n):
            frames.append(draw_frame(mode, i, s, False))
            durations.append(90 if s < n else 220)
    frames.append(draw_frame(mode, K, 99, False)); durations.append(600)
    frames.append(draw_frame(mode, K, 99, True)); durations.append(3200)
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(OUT / f"pipeline_{mode}.gif", save_all=True, append_images=pal[1:],
                duration=durations, loop=0, optimize=True)


for mode in ("light", "dark"):
    results_chart(mode)
    pipeline_gif(mode)
print("done")
