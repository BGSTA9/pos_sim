"""POS rPPG simulation — end-to-end driver.

Pipeline:
    1. SyntheticDataGenerator builds an N x 3 RGB trace from the
       dichromatic reflection model, with intensity and specular distortions
       deliberately tuned to be ~10x stronger than the cardiac pulse
       (worst-case rPPG scenario).
    2. POSProcessor runs Algorithm 1 (sliding window, temporal
       normalization, orthogonal projection, alpha-tuning, overlap-add).
    3. SignalAnalyzer band-passes the resulting H(t), computes the FFT
       and Welch PSD, and locates the dominant BPM peak.

Three figures are produced and saved to ./figures/:
    fig1_raw_rgb.png        - the messy synthetic RGB signal
    fig2_recovered_pulse.png- recovered H(t) overlaid with ground-truth p(t)
    fig3_spectrum.png       - frequency spectrum with the BPM peak
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make `src` importable both when run from the package root and elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.generator import SyntheticDataGenerator
from src.processor import POSProcessor
from src.analyzer import SignalAnalyzer


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _figdir(base: str) -> str:
    out = os.path.join(base, "figures")
    os.makedirs(out, exist_ok=True)
    return out


def plot_raw_rgb(data: dict, outdir: str) -> str:
    """Figure 1: the messy synthetic RGB trace (and its components)."""
    t = data["t"]
    C = data["C"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    colors = ["#c0392b", "#27ae60", "#2980b9"]
    labels = ["R", "G", "B"]
    for c in range(3):
        axes[0].plot(t, C[:, c], color=colors[c], lw=0.8, label=labels[c])
    axes[0].set_title("Synthetic RGB trace from the dichromatic reflection model "
                      "(noise ≈ 10x pulse)")
    axes[0].set_ylabel("Pixel intensity (a.u.)")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, data["i"], color="#8e44ad", lw=0.9, label="i(t) intensity")
    axes[1].plot(t, data["s"], color="#d35400", lw=0.9, label="s(t) specular")
    axes[1].plot(t, data["p"], color="#16a085", lw=1.2, label="p(t) pulse (GT)")
    axes[1].set_title("Hidden components driving the RGB trace")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Amplitude (unitless)")
    axes[1].legend(loc="upper right", ncol=3)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "fig1_raw_rgb.png")
    fig.savefig(path, dpi=140)
    return path


def plot_recovered_pulse(t: np.ndarray, H_clean: np.ndarray, p_gt: np.ndarray,
                          valid_start: int, outdir: str) -> str:
    """Figure 2: recovered pulse H(t) overlaid with ground truth p(t).

    Each is z-scored so they can be compared on a single axis (the absolute
    amplitudes carry no physical meaning here — only the waveform shape and
    phase do).
    """
    def _z(x: np.ndarray) -> np.ndarray:
        x = x - x.mean()
        std = x.std()
        return x / std if std > 0 else x

    H_z = _z(H_clean[valid_start:])
    p_z = _z(p_gt[valid_start:])

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t[valid_start:], p_z, color="#16a085", lw=1.6, alpha=0.85,
            label="Ground-truth pulse p(t) (z-scored)")
    ax.plot(t[valid_start:], H_z, color="#c0392b", lw=1.2, alpha=0.85,
            label="Recovered POS pulse H(t) (band-passed, z-scored)")
    ax.set_title("Recovered POS pulse vs. ground-truth heartbeat")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude (z-score)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_recovered_pulse.png")
    fig.savefig(path, dpi=140)
    return path


def plot_spectrum(result: dict, true_bpm: float, outdir: str) -> str:
    """Figure 3: PSD with the dominant cardiac peak marked."""
    f = result["freqs"]
    P = result["spectrum"]
    bpm = result["bpm"]
    peak_hz = result["peak_hz"]
    method = result["method"].upper()

    bpm_axis = f * 60.0
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(bpm_axis, P, color="#2c3e50", lw=1.2)
    ax.axvline(bpm, color="#c0392b", ls="--", lw=1.2,
               label=f"Detected peak: {bpm:.2f} BPM ({peak_hz:.3f} Hz)")
    ax.axvline(true_bpm, color="#16a085", ls=":", lw=1.4,
               label=f"Ground truth: {true_bpm:.2f} BPM")
    ax.set_xlim(0, min(bpm_axis.max(), 300))
    ax.set_xlabel("Frequency [BPM]")
    ax.set_ylabel(f"{method} spectrum (power)")
    ax.set_title("Spectrum of the recovered pulse with the dominant peak")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_spectrum.png")
    fig.savefig(path, dpi=140)
    return path


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run(
    fs: float = 30.0,
    duration_s: float = 30.0,
    pulse_bpm: float = 72.0,
    noise_ratio: float = 10.0,
    seed: int = 42,
    show: bool = False,
) -> dict:
    """Run the full simulation and return a results dict."""
    gen = SyntheticDataGenerator(
        fs=fs,
        duration_s=duration_s,
        pulse_bpm=pulse_bpm,
        noise_to_pulse_ratio=noise_ratio,
        rng=seed,
    )
    data = gen.generate()

    proc = POSProcessor(fs=fs)
    H = proc.process(data["C"])

    analyzer = SignalAnalyzer(fs=fs)
    H_bp = analyzer.bandpass(H)
    bpm_result = analyzer.estimate_bpm(H_bp, method="welch")
    comparison = analyzer.compare_to_ground_truth(
        bpm_result["bpm"], true_bpm=pulse_bpm
    )

    # The first (l-1) samples of H received no full-window contribution; mark
    # them invalid so plots and metrics ignore them.
    valid_start = proc.window_samples - 1

    outdir = _figdir(_HERE)
    f1 = plot_raw_rgb(data, outdir)
    f2 = plot_recovered_pulse(
        data["t"], H_bp, data["p"], valid_start, outdir
    )
    f3 = plot_spectrum(bpm_result, true_bpm=pulse_bpm, outdir=outdir)

    print("=" * 60)
    print(" POS rPPG simulation — results")
    print("=" * 60)
    print(f"  fs               : {fs} Hz")
    print(f"  duration         : {duration_s} s  ({data['C'].shape[0]} samples)")
    print(f"  window length l  : {proc.window_samples} frames "
          f"({proc.window_samples / fs:.2f} s)")
    print(f"  noise-to-pulse   : {noise_ratio:.1f}x")
    print(f"  ground truth     : {pulse_bpm:.2f} BPM "
          f"({pulse_bpm / 60.0:.3f} Hz)")
    print(f"  detected BPM     : {comparison['estimated_bpm']:.2f}")
    print(f"  abs error        : {comparison['abs_error_bpm']:.2f} BPM "
          f"({comparison['rel_error_pct']:+.2f} %)")
    print("-" * 60)
    print(f"  figure 1 -> {f1}")
    print(f"  figure 2 -> {f2}")
    print(f"  figure 3 -> {f3}")
    print("=" * 60)

    if show:
        plt.show()
    else:
        plt.close("all")

    return {
        "data": data,
        "H": H,
        "H_bp": H_bp,
        "bpm_result": bpm_result,
        "comparison": comparison,
        "figures": (f1, f2, f3),
    }


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fs", type=float, default=30.0, help="frame rate (Hz)")
    p.add_argument("--duration", type=float, default=30.0, help="duration (s)")
    p.add_argument("--bpm", type=float, default=72.0, help="ground-truth BPM")
    p.add_argument("--noise", type=float, default=10.0,
                   help="noise-to-pulse amplitude ratio")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--show", action="store_true",
                   help="open figures interactively in addition to saving")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    run(
        fs=args.fs,
        duration_s=args.duration,
        pulse_bpm=args.bpm,
        noise_ratio=args.noise,
        seed=args.seed,
        show=args.show,
    )
