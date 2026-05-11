"""Signal post-processing and BPM estimation for POS output.

This module is the "grader": it takes the raw POS pulse signal H(t),
band-passes it to the physiological heart-rate band, and reads off the
dominant frequency via FFT and Welch's PSD.

Cutoffs
-------
The default 0.7-4.0 Hz band corresponds to 42-240 BPM (per the prompt),
which spans rest, exercise, infants and the recovery transient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, welch


@dataclass
class SignalAnalyzer:
    """Band-pass, FFT/Welch, BPM peak picking, ground-truth comparison.

    Parameters
    ----------
    fs : float
        Sampling rate of the input signal (Hz).
    low_hz, high_hz : float
        Butterworth bandpass cutoffs in Hz. Default 0.7 Hz (42 BPM) to
        4.0 Hz (240 BPM).
    order : int
        Order per direction of the Butterworth filter. `filtfilt` applies it
        twice (zero-phase), so effective order is 2 * `order`. Default 4.
    welch_seconds : float
        Welch segment length in seconds. Longer = finer frequency resolution
        but fewer averages. Default 8 s.
    """

    fs: float
    low_hz: float = 0.7
    high_hz: float = 4.0
    order: int = 4
    welch_seconds: float = 8.0

    def __post_init__(self) -> None:
        nyq = 0.5 * self.fs
        if not (0 < self.low_hz < self.high_hz < nyq):
            raise ValueError(
                f"Invalid bandpass: need 0 < {self.low_hz} < {self.high_hz} "
                f"< nyquist ({nyq})."
            )
        self._b, self._a = butter(
            self.order,
            [self.low_hz / nyq, self.high_hz / nyq],
            btype="bandpass",
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def bandpass(self, x: np.ndarray) -> np.ndarray:
        """Zero-phase Butterworth bandpass via `scipy.signal.filtfilt`."""
        x = np.asarray(x, dtype=float)
        # filtfilt needs padlen < len(x); for short traces shrink it gracefully.
        padlen = min(3 * max(len(self._a), len(self._b)), len(x) - 1)
        return filtfilt(self._b, self._a, x, padlen=padlen)

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def fft_spectrum(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One-sided magnitude spectrum.

        Returns (freqs_hz, magnitude). Length is len(x)//2 + 1.
        """
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        # Hann-window first to leak less into neighbouring bins.
        w = np.hanning(N)
        X = np.fft.rfft(x * w)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
        return freqs, np.abs(X)

    def welch_psd(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Welch's PSD with a `welch_seconds`-long Hann segment, 50% overlap."""
        x = np.asarray(x, dtype=float)
        nperseg = min(int(round(self.welch_seconds * self.fs)), len(x))
        noverlap = nperseg // 2
        freqs, psd = welch(
            x,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
        )
        return freqs, psd

    # ------------------------------------------------------------------
    # Peak picking
    # ------------------------------------------------------------------

    @staticmethod
    def _parabolic_interp(y_m1: float, y_0: float, y_p1: float) -> float:
        """Quadratic peak interpolation around 3 equispaced samples.

        Given y[k-1], y[k], y[k+1] with y[k] the maximum, returns the
        sub-bin offset `delta` in (-0.5, 0.5) of the true peak relative to k.
        Standard textbook trick (Smith, "Spectral Audio Signal Processing").
        """
        denom = (y_m1 - 2.0 * y_0 + y_p1)
        if denom == 0.0:
            return 0.0
        return 0.5 * (y_m1 - y_p1) / denom

    def estimate_bpm(
        self,
        x: np.ndarray,
        method: str = "welch",
        search_low_hz: Optional[float] = None,
        search_high_hz: Optional[float] = None,
        interpolate: bool = True,
    ) -> dict:
        """Locate the dominant cardiac peak inside [search_low_hz, search_high_hz].

        Parameters
        ----------
        x : (N,) array
            Cleaned pulse signal (typically the output of `bandpass`).
        method : {'welch', 'fft'}
            Which spectral estimator to peak-pick. Welch is more robust.
        search_low_hz, search_high_hz : float, optional
            Restrict the search; defaults to the analyzer's bandpass band.
        interpolate : bool
            If True, refine the peak location with parabolic interpolation
            around the dominant bin. This recovers sub-bin accuracy, which
            matters for Welch where the bin width can be several BPM.

        Returns
        -------
        dict with keys 'bpm', 'peak_hz', 'peak_power', 'freqs', 'spectrum'.
        """
        lo = self.low_hz if search_low_hz is None else search_low_hz
        hi = self.high_hz if search_high_hz is None else search_high_hz

        if method == "welch":
            freqs, spectrum = self.welch_psd(x)
        elif method == "fft":
            freqs, spectrum = self.fft_spectrum(x)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        band = (freqs >= lo) & (freqs <= hi)
        if not np.any(band):
            raise ValueError("Search band is empty; check fs and cutoffs.")
        band_idx = np.where(band)[0]
        rel_peak = int(np.argmax(spectrum[band_idx]))
        peak_idx = band_idx[rel_peak]

        # Parabolic interpolation for sub-bin accuracy. Skip at array edges
        # where one of the neighbours doesn't exist.
        if interpolate and 0 < peak_idx < len(spectrum) - 1:
            delta = self._parabolic_interp(
                float(spectrum[peak_idx - 1]),
                float(spectrum[peak_idx]),
                float(spectrum[peak_idx + 1]),
            )
        else:
            delta = 0.0
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
        peak_hz = float(freqs[peak_idx]) + delta * df
        bpm = peak_hz * 60.0

        return {
            "bpm": bpm,
            "peak_hz": peak_hz,
            "peak_power": float(spectrum[peak_idx]),
            "freqs": freqs,
            "spectrum": spectrum,
            "method": method,
            "interp_delta_bins": float(delta),
        }

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def compare_to_ground_truth(estimated_bpm: float, true_bpm: float) -> dict:
        """Absolute and relative BPM error vs. the ground truth."""
        err = estimated_bpm - true_bpm
        return {
            "estimated_bpm": float(estimated_bpm),
            "true_bpm": float(true_bpm),
            "abs_error_bpm": float(abs(err)),
            "signed_error_bpm": float(err),
            "rel_error_pct": float(100.0 * err / true_bpm) if true_bpm else float("nan"),
        }
