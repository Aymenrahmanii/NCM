"""Animated GIFs for the README, drawn from result files (CPU only, a few seconds).

    python scripts/make_readme_gifs.py

- docs/assets/coarse_to_fine.gif: the toy NCM tokenizer rebuilding a real test story from
  its first 1, 2, ..., 6 levels (text from results/phase2_level_by_level.txt, accuracy from
  results/phase2_recon_by_level.csv).
- docs/assets/generation.gif: the same story written by an LLM (one token per pass) and by
  NCM (one level per pass, real tokenizer outputs). Pass count only, not time.
(docs/assets/architecture.gif is drawn by scripts/make_architecture_gif.py.)
"""

from __future__ import annotations

import csv
import io
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

RESULTS = Path("results")
OUT = Path("docs/assets")
LEVELS = (1, 4, 8, 16, 32, 64)
COLORS = ["#3b0f70", "#6a1b9a", "#8e44ad", "#c0392b", "#e67e22", "#f1c40f"]
INK, MUTED, EMPTY = "#222222", "#777777", "#eeeeee"


def text_color(k: int) -> str:
    """Level colour for text; the last two levels' yellow/orange is too light on white."""
    return COLORS[k] if k < 4 else "#b9770e"


def frame(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").quantize(colors=64)


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB, {len(frames)} frames)")


def draw_blocks(ax, y: float, n: int, filled: int, color: str, label: str) -> None:
    """The 64-token canvas as n equal blocks, the first `filled` of them coloured."""
    w = 64 / n
    for j in range(n):
        ax.add_patch(Rectangle((j * w, y), w, 0.9, facecolor=color if j < filled else EMPTY,
                               edgecolor="white", lw=1.5))
    ax.text(-1, y + 0.45, label, ha="right", va="center", fontsize=10, color=INK)


def story_levels() -> tuple[str, list[str]]:
    text = (RESULTS / "phase2_level_by_level.txt").read_text(encoding="utf-8")
    block = text.split("== real story 1")[1].split("== real story 2")[0]
    original = re.search(r"original: (.*?)\nlevel 1:", block, re.S).group(1)
    levels = re.findall(r"level \d: (.*?)(?=\nlevel \d: |\Z)", block, re.S)
    clean = [" ".join(s.split()) for s in [original, *levels]]
    return clean[0], clean[1:]


def coarse_to_fine() -> None:
    original, levels = story_levels()
    with open(RESULTS / "phase2_recon_by_level.csv", newline="") as f:
        acc = [float(r["token_accuracy"]) for r in csv.DictReader(f)]
    frames, durations = [], []
    for k in range(len(LEVELS)):
        fig = plt.figure(figsize=(9, 4.2))
        ax = fig.add_axes([0.12, 0.62, 0.86, 0.22])
        ax.set_xlim(-0.5, 64.5)
        ax.set_ylim(-0.2, 1.2)
        ax.axis("off")
        n = LEVELS[k]
        for j in range(n):  # one block per code of this level
            w = 64 / n
            ax.add_patch(Rectangle((j * w, 0), w - 0.25, 0.9, color=COLORS[k], lw=0))
        ax.text(-1, 0.45, f"level {k + 1}\n{n} code{'s' * (n > 1)}", ha="right", va="center",
                fontsize=10, color=INK)
        fig.text(0.5, 0.94, "Coarse to fine: the toy NCM rebuilding a real story from its "
                 "first levels", ha="center", fontsize=12, weight="bold", color=INK)
        fig.text(0.5, 0.88, f"levels 1–{k + 1} used · token accuracy {acc[k]:.0%}",
                 ha="center", fontsize=10, color=COLORS[k] if k < 4 else "#b9770e")
        shown = levels[k][:230] + ("…" if len(levels[k]) > 230 else "")
        fig.text(0.06, 0.52, "decoded:", fontsize=9, color=MUTED)
        fig.text(0.06, 0.48, "\n".join(textwrap.wrap(shown, 95)), fontsize=9.5, va="top",
                 family="monospace", color=INK)
        fig.text(0.06, 0.16, "original:", fontsize=9, color=MUTED)
        fig.text(0.06, 0.12, "\n".join(textwrap.wrap(original[:190] + "…", 95)), fontsize=9,
                 va="top", family="monospace", color=MUTED)
        frames.append(frame(fig))
        durations.append(3200 if k == len(LEVELS) - 1 else 1500)
    save_gif(frames, durations, OUT / "coarse_to_fine.gif")


def _bar(fig, y: float, n: int, filled: int, color: str) -> None:
    bar = fig.add_axes([0.06, y, 0.88, 0.045])
    bar.set_xlim(0, 64)
    bar.set_ylim(0, 0.9)
    bar.axis("off")
    draw_blocks(bar, 0, n, filled, color, "")


def generation() -> None:
    """Same story, two writing orders, synced on forward passes (NCM's 6 spread over 64)."""
    original, levels = story_levels()
    words = original[:260].split()
    frames, durations = [], []
    for p in range(0, 65, 2):
        k = int(p * len(LEVELS) / 64) - 1  # NCM level finished by AR pass p
        fig = plt.figure(figsize=(9, 4.4))
        fig.text(0.5, 0.95, "How they write the same 64-token story", ha="center",
                 fontsize=13, weight="bold", color=INK)
        fig.text(0.06, 0.85, f"LLM (autoregressive) · pass {p}/64 · one token per pass, "
                 "left to right", fontsize=10.5, weight="bold", color="#b9770e")
        _bar(fig, 0.77, 64, p, COLORS[5])
        shown = " ".join(words[: round(len(words) * p / 64)]) + ("▌" if p < 64 else "")
        fig.text(0.06, 0.72, "\n".join(textwrap.wrap(shown, 92)), fontsize=9.5, va="top",
                 family="monospace", color=INK)

        n = LEVELS[max(k, 0)]
        codes = f"{n} code{'s' * (n > 1)}"
        label = (f"NCM · pass {k + 1}/6 · level {k + 1}: {codes} for the whole text, in "
                 "parallel" if k >= 0 else "NCM · pass 0/6 · empty canvas")
        fig.text(0.06, 0.53, label, fontsize=10.5, weight="bold",
                 color=text_color(k) if k >= 0 else INK)
        _bar(fig, 0.45, n, n if k >= 0 else 0, COLORS[max(k, 0)])
        text = levels[k][:260] if k >= 0 else ""
        fig.text(0.06, 0.40, "\n".join(textwrap.wrap(text, 92)), fontsize=9.5, va="top",
                 family="monospace", color=INK)
        fig.text(0.5, 0.03, "NCM text is the toy tokenizer's real output from levels 1..k. "
                 "Pass count, not time: each NCM pass works\non the whole canvas, and on an "
                 "RTX 3050 the official checkpoints roughly tie with same-size AR models at "
                 "batch 1.", ha="center", fontsize=8.5, color=MUTED)
        frames.append(frame(fig))
        durations.append(3000 if p == 64 else 110)
    save_gif(frames, durations, OUT / "generation.gif")


if __name__ == "__main__":
    coarse_to_fine()
    generation()
