"""POS (Plane-Orthogonal-to-Skin) rPPG algorithm.

Implements Algorithm 1 of Wang, den Brinker, Stuijk, de Haan (2017).

Pseudocode (paper):

    Initialize: H = zeros(1, N), l = 32   (at 20 fps)
    for n = 1, 2, ..., N do
        C(n) = [R(n), G(n), B(n)]^T              # spatial averaging
        if m = n - l + 1 > 0 then
            C_n^i = C_{m..n}^i / mean(C_{m..n}^i)        # temporal normalization
            S     = [[0, 1, -1], [-2, 1, 1]] * C_n       # projection
            h     = S_1 + (sigma(S_1) / sigma(S_2)) * S_2  # alpha-tuning
            H_{m..n} = H_{m..n} + (h - mean(h))          # overlap-adding
        end if
    end for
    Output: H

The math at a glance
--------------------
The POS projection matrix P = [[0,1,-1], [-2,1,1]] has two essential
properties:

(1) P @ [1,1,1]^T = [0,0]^T, i.e. its row-space is orthogonal to the
    temporally-normalized skin-tone direction `1`. This kills the common-mode
    intensity term i(t) (Eq. (12) of the paper) at first order.

(2) Acting on a temporally-normalized trace `C_n`:
        S_1 = G_n - B_n
        S_2 = G_n + B_n - 2 R_n
    For a blood-volume vector u_pbv with the standard G > B > R ranking,
    *both* S_1 and S_2 carry the pulse with the same sign — they are in-phase
    on the pulse and (typically) anti-phase on the residual specular noise.
    That is why alpha-tuning uses `+` here, vs. CHROM's `-`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# The POS projection matrix from Eq. (33) of Wang et al. (2017).
P_POS: np.ndarray = np.array(
    [[0.0,  1.0, -1.0],
     [-2.0, 1.0,  1.0]],
    dtype=float,
)


@dataclass
class POSProcessor:
    """Sliding-window POS pulse extractor.

    Parameters
    ----------
    fs : float
        Sampling frequency of the input RGB trace (Hz).
    window_seconds : float, optional
        Length of the sliding window in seconds. If `window_samples` is also
        given, it takes priority. Default 1.6 s — about two cardiac cycles
        at 72 BPM, mirroring the paper's "32 frames @ 20 fps" choice.
    window_samples : int, optional
        Explicit window length l in frames. Overrides `window_seconds`.
    eps : float
        Numerical floor for the per-window mean and for sigma(S_2) in
        alpha-tuning. Guards against divide-by-zero in pathological inputs.
    """

    fs: float
    window_seconds: float = 1.6
    window_samples: Optional[int] = None
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.window_samples is None:
            self.window_samples = int(round(self.window_seconds * self.fs))
        if self.window_samples < 4:
            raise ValueError(
                f"window_samples = {self.window_samples} is too small; "
                "must cover at least one cardiac cycle."
            )

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def process(self, C: np.ndarray) -> np.ndarray:
        """Run POS on an N x 3 RGB trace and return the pulse signal H (N,).

        Parameters
        ----------
        C : (N, 3) array of float
            Spatially-averaged RGB trace, ordered as [R, G, B] columns.

        Returns
        -------
        H : (N,) array of float
            The overlap-added pulse signal. Samples in [0, l-1) get no
            contribution (the algorithm needs at least one full window) and
            are returned as 0 — they can be safely discarded by the caller.
        """
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[1] != 3:
            raise ValueError(f"C must have shape (N, 3); got {C.shape}")

        N = C.shape[0]
        l = self.window_samples
        H = np.zeros(N, dtype=float)

        if N < l:
            # Not enough samples for a single window; nothing to do.
            return H

        # ---- Vectorised implementation ------------------------------------
        # We construct the (num_windows, l, 3) stack of overlapping windows
        # using a stride trick so the per-window work below is pure NumPy.
        # num_windows = N - l + 1, and window k spans samples [k, k + l).
        stride_bytes = C.strides[0]
        num_windows = N - l + 1
        windows = np.lib.stride_tricks.as_strided(
            C,
            shape=(num_windows, l, 3),
            strides=(stride_bytes, stride_bytes, C.strides[1]),
            writeable=False,
        )

        # Step 1: temporal normalization (Eq. (11)).
        # For each window divide each channel by its temporal mean so the
        # static skin-tone collapses to [1, 1, 1].
        means = windows.mean(axis=1, keepdims=True)               # (W, 1, 3)
        # Safety against degenerate channels (e.g. all-zero trace in tests).
        means = np.where(np.abs(means) < self.eps, self.eps, means)
        Cn = windows / means                                      # (W, l, 3)

        # Step 2: orthogonal projection. Apply P to the RGB axis -> 2 signals.
        # Cn @ P.T gives (W, l, 2) where [..., 0] = S_1 and [..., 1] = S_2.
        S = Cn @ P_POS.T                                          # (W, l, 2)
        S1 = S[..., 0]                                            # (W, l)
        S2 = S[..., 1]                                            # (W, l)

        # Step 3: alpha-tuning. alpha = sigma(S_1) / sigma(S_2) (per window).
        # ddof=0 matches the paper's std-as-RMS-of-zero-mean convention.
        sigma1 = S1.std(axis=1, ddof=0)                           # (W,)
        sigma2 = S2.std(axis=1, ddof=0)                           # (W,)
        alpha = sigma1 / np.maximum(sigma2, self.eps)             # (W,)

        # Step 4: signal combination h = S_1 + alpha * S_2.
        h = S1 + alpha[:, None] * S2                              # (W, l)

        # Step 5: zero-mean each window, then overlap-add into H.
        h = h - h.mean(axis=1, keepdims=True)                     # (W, l)

        # Overlap-add: window k contributes to H[k : k+l]. The naive Python
        # loop is W passes of length-l adds; that is O(N) in total and
        # plenty fast for video-rate traces, so we keep it explicit and
        # legible rather than reaching for fancy ufuncs.
        for k in range(num_windows):
            H[k:k + l] += h[k]

        return H
