"""Generate the project architecture diagram (figures/architecture.png).

Single-page, top-to-bottom pipeline view:

   Inputs -> Class 1 (generator) -> RGB trace
          -> Class 2 (POS processor) -> H(t)
          -> Class 3 (analyzer) -> BPM
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Palette — one muted pastel per pipeline stage.
C_INPUT  = "#dde7f2"
C_CLASS1 = "#fff1c9"
C_CLASS2 = "#ffd9d0"
C_CLASS3 = "#d6efd6"
C_DATA   = "#eeeeee"
C_OUTPUT = "#ffd066"

EDGE     = "#333333"
ARROW_C  = "#222222"


def _box(ax, x, y, w, h, title, body, fc,
         title_size=13, body_size=10):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.25",
        facecolor=fc, edgecolor=EDGE, lw=1.6, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h - 0.30, title,
        ha="center", va="top",
        fontsize=title_size, fontweight="bold", zorder=3,
    )
    ax.text(
        x + 0.30, y + h - 0.95, body,
        ha="left", va="top",
        fontsize=body_size, family="DejaVu Sans Mono", zorder=3,
    )


def _pill(ax, x, y, w, h, text, fc=C_DATA, fs=11, italic=True, bold=False):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.55",
        facecolor=fc, edgecolor="#888", lw=1.0, zorder=2,
    )
    ax.add_patch(rect)
    style = "italic" if italic else "normal"
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fs, style=style, fontweight=weight, zorder=3,
    )


def _arrow(ax, x, y_from, y_to, label=""):
    a = FancyArrowPatch(
        (x, y_from), (x, y_to),
        arrowstyle="-|>", mutation_scale=22,
        lw=2.0, color=ARROW_C, zorder=2,
    )
    ax.add_patch(a)
    if label:
        ax.text(
            x + 0.18, (y_from + y_to) / 2, label,
            ha="left", va="center",
            fontsize=10, style="italic", color="#444", zorder=3,
        )


def build(outpath: str) -> str:
    fig = plt.figure(figsize=(11.5, 18.0))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 26)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Title -------------------------------------------------------
    ax.text(5.5, 25.0, "POS rPPG — End-to-End Pipeline",
            ha="center", fontsize=20, fontweight="bold")
    ax.text(5.5, 24.3,
            "From synthesized skin physics  →  recovered heart rate",
            ha="center", fontsize=12, style="italic", color="#555")

    # ---- INPUTS ------------------------------------------------------
    _pill(ax, 2.0, 22.7, 7.0, 0.9,
          "Inputs:  fs (frame rate),  duration,  ground-truth BPM,  RNG seed",
          fc=C_INPUT, fs=11)

    _arrow(ax, 5.5, 22.7, 21.7)

    # ---- CLASS 1 -----------------------------------------------------
    _box(ax, 0.5, 17.0, 10.0, 4.7,
         "Class 1 — SyntheticDataGenerator     (the fake face)",
         "Builds an N x 3 RGB trace from the dichromatic reflection model\n"
         "(Eq. 6 of Wang et al. 2017):\n"
         "\n"
         "    C(t) = I0 (1+i(t)) ( u_c·c0 + u_s·s(t) + u_p·p(t) ) + v_n(t)\n"
         "\n"
         "      u_c·c0   skin baseline   [0.77, 0.51, 0.38]\n"
         "      u_p·p(t) cardiac pulse   G > B > R   (1.2 Hz sine)\n"
         "      i(t)     intensity drift (shadows / motion)\n"
         "      u_s·s(t) specular drift  (glints from lamp)\n"
         "      v_n(t)   white Gaussian sensor noise\n"
         "\n"
         "  Worst-case knob:  i(t) and s(t) ~ 10x pulse amplitude per channel.",
         fc=C_CLASS1)

    _arrow(ax, 5.5, 17.0, 16.0, label="  N x 3 RGB trace")

    _pill(ax, 1.5, 14.7, 8.0, 1.1,
          "Messy RGB trace  —  pulse hidden at ~ 0.1 % modulation",
          fc=C_DATA, fs=11)

    _arrow(ax, 5.5, 14.7, 13.8)

    # ---- CLASS 2 -----------------------------------------------------
    _box(ax, 0.5, 7.3, 10.0, 6.5,
         "Class 2 — POSProcessor   (Algorithm 1 — the de-mixer)",
         "Sliding window of length  l ≈ 1.6 · fs  (~ 2 cardiac cycles, 1-frame hop).\n"
         "For each window:\n"
         "\n"
         "    Step 1   Temporal normalization\n"
         "               C_n = C / mean(C)        → maps skin tone to (1,1,1)\n"
         "\n"
         "    Step 2   Orthogonal projection (kills common-mode intensity)\n"
         "               S  = P · C_n        P = [[ 0, 1, -1],\n"
         "                                        [-2, 1,  1]]\n"
         "               S1 = Gn − Bn\n"
         "               S2 = Gn + Bn − 2 Rn\n"
         "\n"
         "    Step 3   Alpha-tuning             α = σ(S1) / σ(S2)\n"
         "    Step 4   Pulse combination        h = S1 + α · S2\n"
         "    Step 5   Overlap-add zero-mean h  into  H(t)",
         fc=C_CLASS2)

    _arrow(ax, 5.5, 7.3, 6.3, label="  H(t)  pulse signal")

    _pill(ax, 1.5, 5.0, 8.0, 1.1,
          "Recovered pulse H(t)  —  single channel, in heart-rate band",
          fc=C_DATA, fs=11)

    _arrow(ax, 5.5, 5.0, 4.1)

    # ---- CLASS 3 -----------------------------------------------------
    _box(ax, 0.5, 1.0, 10.0, 3.1,
         "Class 3 — SignalAnalyzer    (the reader)",
         "  • Zero-phase Butterworth bandpass    0.7 – 4.0 Hz   (42–240 BPM)\n"
         "  • Spectrum    FFT  +  Welch PSD  (Hann window, 50 % overlap)\n"
         "  • Peak picking with parabolic interpolation (sub-bin BPM)",
         fc=C_CLASS3)

    _arrow(ax, 5.5, 1.0, 0.1)

    # ---- OUTPUT ------------------------------------------------------
    _pill(ax, 3.0, -1.1, 5.0, 1.1,
          "Output:  BPM ≈ 72.7   (|err| ~ 0.7 BPM)",
          fc=C_OUTPUT, fs=13, italic=False, bold=True)

    # Re-extend the y-limits to include the output pill below 0.
    ax.set_ylim(-1.6, 26)

    # ---- Footer ------------------------------------------------------
    ax.text(0.5, -1.5,
            "Maps 1:1 to the code in pos_sim/src/  "
            "(generator.py, processor.py, analyzer.py)  "
            "and the driver pos_sim/main.py.",
            ha="left", va="top", fontsize=9, color="#666")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "figures", "architecture.png")
    path = build(out)
    print(f"Wrote {os.path.abspath(path)}")
