"""Data-flow animation of the generation loop of an LLM and of NCM, from the real toy models.

    python scripts/make_architecture_gif.py            # draw (from the saved JSON)
    python scripts/make_architecture_gif.py --extract  # rerun the Phase 2 toy models first

Step 1 (--extract, needs artifacts/toy_ncm/ and data/tinystories/ from phase2-train) runs
the toy AR model, tokenizer and generator on CPU on the first test story and saves what the
animation shows to results/architecture_anim_data.json: the AR model's top-5 next tokens,
the NCM canvas after each level, each level's codes and the generator's probability for
them (teacher-forced: the real story's codes), and the text decoded after each level.
Step 2 draws docs/assets/architecture.gif from that file only.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

DATA = Path("results/architecture_anim_data.json")
OUT = Path("docs/assets/architecture.gif")
LEVELS = (1, 4, 8, 16, 32, 64)
COLORS = ["#3b0f70", "#6a1b9a", "#8e44ad", "#c0392b", "#e67e22", "#f1c40f"]
AR_COLOR, AR_INK = "#e0a800", "#b9770e"
INK, MUTED = "#222222", "#777777"
AR_POSITIONS = range(8, 14)  # predict tokens 9..14 of the story (one per pass shown)
FRAMES_PER_STEP = 8


# ---- step 1: run the toy models ------------------------------------------------------------

def extract() -> None:
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "src")
    from ncm_lab.toy_ncm.ar_baseline import load_ar
    from ncm_lab.toy_ncm.config import EOS_ID
    from ncm_lab.toy_ncm.data import decode_ids, load_bpe, load_tokens
    from ncm_lab.toy_ncm.generator import load_generator
    from ncm_lab.toy_ncm.tokenizer import down, load_tokenizer, up

    torch.manual_seed(0)
    bpe = load_bpe()
    _, test = load_tokens()
    ids = test[:1]
    tok, gen, ar = load_tokenizer("cpu"), load_generator("cpu"), load_ar("cpu")
    piece = [bpe.decode([int(i)]) for i in ids[0]]
    out: dict = {"tokens": piece, "ar": [], "ncm": []}
    with torch.no_grad():
        inp = torch.cat([torch.full_like(ids[:, :1], EOS_ID), ids[:, :-1]], dim=1)
        probs = F.softmax(ar(inp)[0].float(), -1)
        for t in AR_POSITIONS:
            p = probs[t]
            top = torch.topk(p, 5)
            real = int(ids[0, t])
            out["ar"].append({
                "position": t, "top": [[bpe.decode([int(i)]), round(float(v), 3)]
                                       for v, i in zip(top.values, top.indices)],
                "real": piece[t], "p_real": round(float(p[real]), 3),
                "rank_real": int((p > p[real]).sum()) + 1})

        codes, f_hats, _ = tok.quantize(tok.encode(ids))
        before = [torch.zeros_like(f_hats[0])] + f_hats[:-1]
        logits = gen(before)
        for k, n in enumerate(LEVELS):
            pk = F.softmax(logits[k][0].float(), -1)
            c = codes[k][0]
            inc = up(tok.quantizers[k].lookup(codes[k]))[0]
            out["ncm"].append({
                "level": k + 1, "length": n,
                "canvas_before": before[k][0].numpy().round(3).tolist(),
                "down": down(before[k], n)[0].numpy().round(3).tolist(),
                "codes": c.tolist(),
                "p_real": [round(float(pk[i, c[i]]), 3) for i in range(n)],
                "top1": (pk.argmax(-1) == c).tolist(),
                "increment": inc.numpy().round(3).tolist(),
                "canvas_after": f_hats[k][0].numpy().round(3).tolist(),
                "text": decode_ids(bpe, tok.decode(f_hats[k]).argmax(-1)[0].tolist())})
    DATA.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {DATA} ({DATA.stat().st_size / 1e3:.0f} kB)")


# ---- step 2: draw ------------------------------------------------------------------------

def _frame(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").quantize(colors=96)


def _box(ax, x, y, w, h, title, active, color) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor="white", edgecolor=color,
                                lw=2.6 if active else 1.0, alpha=1.0))
    ax.text(x + w / 2, y + h + 0.06, title, ha="center", va="bottom", fontsize=8.5,
            color=color if active else MUTED, weight="bold" if active else "normal")


def _heat(ax, arr, x, y, w, h, vmax, alpha=1.0) -> None:
    """Draw a (positions, dims) array as a dims x positions heatmap in data coords."""
    a = np.asarray(arr, dtype=float).T
    ax.imshow(a, extent=(x, x + w, y, y + h), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              aspect="auto", interpolation="nearest", alpha=alpha, zorder=3)


def _arcs(ax, src: list[float], dst: float, y: float, color: str, alpha: float,
          height: float = 0.5) -> None:
    for x in src:
        ax.add_patch(FancyArrowPatch((x, y), (dst, y), arrowstyle="-",
                                     connectionstyle=f"arc3,rad={-height}", lw=0.9,
                                     color=color, alpha=alpha, zorder=4))


def _llm_panel(ax, d: dict, step: int, sub: int) -> None:
    """One AR pass = predict token t from tokens < t. sub: 0..FRAMES_PER_STEP-1."""
    a = d["ar"][step]
    t = a["position"]
    words = d["tokens"]
    ax.text(0.0, 4.75, f"LLM (autoregressive) · pass {t + 1}/64: predict the next token",
            fontsize=11, weight="bold", color=AR_INK)
    # context strip: the last 10 tokens before t, as chips
    shown = list(range(max(0, t - 9), t))
    added = sub >= 7
    chips = shown + ([t] if added else [])
    xs = {}
    for j, i in enumerate(chips):
        x = 0.1 + j * 0.62
        xs[i] = x + 0.29
        new = i == t
        ax.add_patch(FancyBboxPatch((x, 3.55), 0.56, 0.42, boxstyle="round,pad=0.02",
                                    facecolor=AR_COLOR if new else "#fff6d5",
                                    edgecolor=AR_INK, lw=1.5 if new else 0.6, zorder=3))
        ax.text(x + 0.28, 3.76, words[i].strip() or "␣", ha="center", va="center",
                fontsize=7.5, color=INK, zorder=5)
    ax.text(0.1, 4.15, "tokens so far" + (" (… earlier ones not shown)" if t > 9 else ""),
            fontsize=8, color=MUTED)
    # transformer box with causal attention from the newest position to all earlier ones
    _box(ax, 0.05, 1.2, 6.6, 1.65, "Transformer, causal mask: the newest position attends "
         "to every earlier token", 1 <= sub <= 3, AR_INK)
    cols = [0.1 + j * 0.62 + 0.29 for j in range(len(shown))]
    for x in cols:
        ax.add_patch(Rectangle((x - 0.2, 1.35), 0.4, 0.35, facecolor="#fff6d5",
                               edgecolor=AR_INK, lw=0.5, zorder=3))
    if 1 <= sub <= 6:
        _arcs(ax, cols[:-1], cols[-1], 1.72, AR_INK, alpha=min(1.0, 0.35 * sub), height=0.45)
    if sub >= 1:
        ax.annotate("", xy=(cols[-1], 1.7), xytext=(cols[-1], 3.5),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    # next-token head: real top-5 probabilities of the toy AR model
    _box(ax, 7.2, 1.2, 4.6, 3.0, "next-token head: 4,096-way softmax (top 5)", 4 <= sub,
         AR_INK)
    if sub >= 4:
        for r, (w, p) in enumerate(a["top"]):
            y = 3.65 - r * 0.5
            hit = w == a["real"]
            ax.add_patch(Rectangle((8.6, y), 3.0 * p / max(q for _, q in a["top"]), 0.36,
                                   facecolor=AR_COLOR if hit and sub >= 5 else "#d9d9d9",
                                   zorder=3))
            ax.text(8.5, y + 0.18, (w.strip() or "␣")[:10], ha="right", va="center",
                    fontsize=8, color=INK, weight="bold" if hit and sub >= 5 else "normal")
            ax.text(8.65 + 3.0 * p / max(q for _, q in a["top"]), y + 0.18, f"{p:.2f}",
                    va="center", fontsize=7.5, color=MUTED, zorder=4)
    if sub >= 5:
        msg = (f'story\'s next token "{a["real"].strip()}": p = {a["p_real"]:.2f}, '
               f"rank {a['rank_real']}")
        ax.text(9.5, 1.32, msg, ha="center", fontsize=7.5, color=AR_INK)
    if sub == 6:
        ax.annotate("", xy=(0.1 + len(shown) * 0.62 + 0.29, 3.5), xytext=(8.6, 3.9),
                    arrowprops=dict(arrowstyle="-|>", color=AR_INK, lw=1.4,
                                    connectionstyle="arc3,rad=0.25"))
    ax.text(0.0, 0.75, "append it and run the whole model again: one pass per token, "
            "64 passes for 64 tokens", fontsize=8.5, color=AR_INK, style="italic")


def _ncm_panel(ax, d: dict, k: int, sub: int, vmax: float, final: bool) -> None:
    lv = d["ncm"][k]
    n, color = lv["length"], COLORS[k]
    ink = color if k < 4 else AR_INK
    ax.text(0.0, 4.75, f"NCM · pass {k + 1}/6: level {k + 1} writes {n} code"
            f"{'s' * (n > 1)} for the whole text at once", fontsize=11, weight="bold",
            color=ink)
    # canvas (64 positions x 32 dims), real values
    canvas = lv["canvas_after"] if sub >= 6 else lv["canvas_before"]
    _box(ax, 0.05, 2.75, 3.4, 1.35, "canvas: 64 positions × 32 dims", sub in (0, 6, 7),
         ink)
    _heat(ax, canvas, 0.1, 2.8, 3.3, 1.25, vmax)
    # shrink to level length
    _box(ax, 3.85, 2.75, 1.2, 1.35, f"shrink to {n}", sub == 1, ink)
    if sub >= 1:
        _heat(ax, lv["down"], 3.9, 2.8, 1.1, 1.25, vmax)
    # generator input sequence: all levels so far, block-causal attention
    _box(ax, 0.05, 1.05, 6.6, 1.15, "Transformer, block-causal: level k attends to "
         "levels ≤ k", 2 <= sub <= 3, ink)
    total = sum(LEVELS)
    x0, wseg = 0.2, 6.3
    centres = []
    for j, m in enumerate(LEVELS[: k + 1]):
        start = x0 + wseg * sum(LEVELS[:j]) / total
        width = wseg * m / total
        ax.add_patch(Rectangle((start, 1.12), max(width, 0.04), 0.42,
                               facecolor=COLORS[j], alpha=1.0 if j == k else 0.45,
                               edgecolor="white", lw=0.8, zorder=3))
        if width > 0.3:
            ax.text(start + width / 2, 1.33, f"L{j + 1}: {m}", ha="center", va="center",
                    fontsize=7, color="white" if j < 4 else INK, zorder=5)
        centres.append(start + width / 2)
    ax.text(0.2, 1.62, "input sequence: every level so far", fontsize=7.5, color=MUTED)
    if 2 <= sub <= 4 and k:
        _arcs(ax, centres[:-1], centres[-1], 1.56, ink, alpha=min(1.0, 0.4 * (sub - 1)),
              height=0.35)
    # code head: the level's codes and the generator's probability of each
    _box(ax, 7.2, 2.75, 4.6, 1.35, f"code head: 512-way per position ({n} at once)",
         4 <= sub <= 5, ink)
    if sub >= 4:
        p = np.array(lv["p_real"])
        w = 4.4 / n
        ax.plot([7.3, 11.7], [3.75, 3.75], ls=":", lw=0.8, color=MUTED)
        ax.text(11.72, 3.75, "p=1", fontsize=6.5, color=MUTED, va="center")
        for i, pi in enumerate(p):
            ax.add_patch(Rectangle((7.3 + i * w, 2.85), w * 0.9, 0.9 * pi + 0.02,
                                   facecolor=color, zorder=3))
        ax.text(9.5, 3.88, f"probability of the story's code: mean {p.mean():.2f}, "
                f"top-1 {np.mean(lv['top1']):.0%}", ha="center", fontsize=7.5, color=ink)
    # lookup + stretch, added onto the canvas
    _box(ax, 7.2, 1.05, 4.6, 1.15, "look up codes, stretch to 64, add to canvas",
         sub >= 5, ink)
    if sub >= 5:
        _heat(ax, lv["increment"], 7.3, 1.1, 4.4, 1.0, vmax)
        ax.annotate("", xy=(3.4, 2.85), xytext=(7.25, 1.6),
                    arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.4,
                                    connectionstyle="arc3,rad=-0.2"))
        ax.text(5.3, 2.35, "+", fontsize=18, color=ink, weight="bold", ha="center")
    text = lv["text"] if sub >= 7 else (d["ncm"][k - 1]["text"] if k else "")
    label = "decoder → text so far: " if not final else "decoder → final text: "
    ax.text(0.0, 0.62, label + "\n".join(textwrap.wrap(text[:170], 110)), fontsize=7.8,
            va="top", family="monospace", color=INK)


def draw() -> None:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    # a tighter scale than the final canvas's range, so the faint coarse levels show
    vmax = 0.6 * float(np.percentile(np.abs(np.array(d["ncm"][-1]["canvas_after"])), 98))
    frames, durations = [], []
    for k in range(len(LEVELS)):
        for sub in range(FRAMES_PER_STEP):
            fig = plt.figure(figsize=(12, 8.2))
            fig.text(0.5, 0.975, "What happens in one generation pass: LLM vs NCM "
                     "(real toy models, real story)", ha="center", fontsize=13,
                     weight="bold", color=INK)
            top = fig.add_axes([0.02, 0.52, 0.96, 0.42])
            bottom = fig.add_axes([0.02, 0.05, 0.96, 0.42])
            for ax in (top, bottom):
                ax.set_xlim(-0.1, 12)
                ax.set_ylim(0.05, 5.05)
                ax.axis("off")
            _llm_panel(top, d, k, sub)
            _ncm_panel(bottom, d, k, sub, vmax, final=k == len(LEVELS) - 1 and sub >= 7)
            fig.add_artist(plt.Line2D([0.02, 0.98], [0.495, 0.495], color="#dddddd", lw=1))
            fig.text(0.5, 0.012, "Phase 2 toy models: AR 6.9M; NCM generator 5.6M + "
                     "tokenizer 8.6M. Probabilities are the models' own, for the real "
                     "story's next token / codes. Frames are stages, not equal time.",
                     ha="center", fontsize=8, color=MUTED)
            frames.append(_frame(fig))
            durations.append(4000 if (k, sub) == (5, 7) else 1300 if sub == 7 else 380)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true")
    if p.parse_args().extract or not DATA.exists():
        extract()
    draw()
