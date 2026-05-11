"""Physics-based synthetic RGB trace generator for rPPG validation.

Implements the dichromatic skin-reflection model from Wang et al. (2017),
Eq. (6):

    C(t) = I0 * (1 + i(t)) * (u_c * c0 + u_s * s(t) + u_p * p(t)) + v_n(t)

where
    I0         scalar stationary luminance,
    i(t)       zero-mean intensity fluctuation (motion / flicker / shadow),
    u_c * c0   stationary skin reflection vector (DC color),
    u_s        unit color vector of the light spectrum,
    s(t)       zero-mean specular fluctuation (induced by motion),
    u_p        relative pulsatile strengths in RGB (blood-volume pulse vector),
    p(t)       zero-mean cardiac pulse signal,
    v_n(t)     additive white Gaussian sensor / quantization noise.

The output is the spatially-averaged RGB trace (N x 3 array) that one would
measure from a face ROI in a video. This is the "raw input" that the POS
algorithm consumes.

Design choices (see README for derivations):

* `u_c` defaults to the standardized skin-tone vector `[0.77, 0.51, 0.38]`
  from CHROM (Wang/de Haan 2013), which is also referenced in this paper.
* `u_s` defaults to `[1, 1, 1] / sqrt(3)` (white-light specular).
* `u_p` defaults to the `u_pbv = [0.33, 0.77, 0.53]` blood-volume vector
  measured in [5], which respects the G > B > R pulsatile ranking.
* The "worst-case noise = 10x pulse" knob from the prompt is implemented as
  a multiplier on the per-channel AC amplitude that the intensity and
  specular terms inject into the RGB trace, taking each component's color
  vector into account so the comparison is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SyntheticDataGenerator:
    """Generate an N x 3 RGB trace from the dichromatic reflection model.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz (camera frame rate). Default 30.
    duration_s : float
        Total trace length in seconds. Default 30.
    pulse_bpm : float
        Ground-truth heart rate in beats per minute. Default 72 (1.2 Hz).
    I0 : float
        Stationary luminance level. Acts as global scale; default 100.
    u_c : np.ndarray
        Unit-ish skin-tone vector [R, G, B]. Will NOT be re-normalized so the
        caller can encode realistic intensity differences across channels.
    u_s : np.ndarray
        Unit specular color vector [R, G, B]. Default white light.
    u_p : np.ndarray
        Blood-volume pulse vector [R, G, B] (G > B > R by default).
    c0 : float
        Stationary skin reflection strength (multiplier on u_c).
    pulse_amplitude : float
        Peak amplitude of the unit-less pulse signal p(t). Sub-1% modulation
        is realistic for true PPG; we use 0.01 by default.
    noise_to_pulse_ratio : float
        Target ratio of the dominant non-pulsatile per-channel AC amplitude
        to the pulse per-channel AC amplitude. The prompt asks for ~10.
    sensor_noise_std : float
        Standard deviation of v_n(t), white Gaussian, in the same units as
        the final RGB trace. Default 0.05.
    rng : np.random.Generator or int or None
        Source of randomness for reproducibility.

    Notes
    -----
    The "intensity" term i(t) is built from low-frequency components
    (0.05-0.5 Hz band) to mimic ambient flicker / slow shadow drift.
    The "specular" term s(t) is built from slightly higher-frequency
    motion-band content (0.1-0.8 Hz), which overlaps the pulse band and
    is therefore the harder distortion to remove. Both are zero-mean.
    """

    fs: float = 30.0
    duration_s: float = 30.0
    pulse_bpm: float = 72.0
    I0: float = 100.0
    u_c: np.ndarray = field(
        default_factory=lambda: np.array([0.77, 0.51, 0.38])
    )
    u_s: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    )
    u_p: np.ndarray = field(
        default_factory=lambda: np.array([0.33, 0.77, 0.53])
    )
    c0: float = 1.0
    pulse_amplitude: float = 0.01
    noise_to_pulse_ratio: float = 10.0
    sensor_noise_std: float = 0.05
    rng: Optional[object] = None

    def __post_init__(self) -> None:
        self.u_c = np.asarray(self.u_c, dtype=float)
        self.u_s = np.asarray(self.u_s, dtype=float)
        self.u_p = np.asarray(self.u_p, dtype=float)
        self._rng = np.random.default_rng(self.rng)
        self.N: int = int(round(self.fs * self.duration_s))
        self.t: np.ndarray = np.arange(self.N) / self.fs

        if not (self.u_p[1] > self.u_p[2] > self.u_p[0]):
            raise ValueError(
                "u_p must satisfy the G > B > R pulsatile ranking; "
                f"got u_p = {self.u_p.tolist()}"
            )

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _bandlimited_noise(self, low_hz: float, high_hz: float) -> np.ndarray:
        """Generate a zero-mean, unit-variance band-limited signal of length N.

        Constructed in the frequency domain so the band is exact and
        independent of fs / N — much cleaner than IIR-filtering white noise.
        """
        N = self.N
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
        spectrum = (self._rng.standard_normal(freqs.shape)
                    + 1j * self._rng.standard_normal(freqs.shape))
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        spectrum *= mask
        x = np.fft.irfft(spectrum, n=N)
        x -= x.mean()
        std = x.std()
        if std > 0:
            x /= std
        return x

    def _pulse_signal(self) -> np.ndarray:
        """Pure-sine cardiac pulse p(t) at `pulse_bpm`, amplitude `pulse_amplitude`."""
        f_h = self.pulse_bpm / 60.0
        return self.pulse_amplitude * np.sin(2.0 * np.pi * f_h * self.t)

    # ------------------------------------------------------------------
    # Amplitude calibration for the "10x worse than pulse" requirement
    # ------------------------------------------------------------------

    def _calibrate_amplitudes(self) -> tuple[float, float]:
        """Solve for the std of i(t) and s(t) such that each distortion's
        per-channel AC amplitude is `noise_to_pulse_ratio` x the pulse's.

        From Eq. (6), keeping only the AC terms and dropping the (negligible)
        AC-AC products:

            pulse AC, channel c        ~  I0 * u_p[c] * std(p)
            intensity AC, channel c    ~  I0 * (u_c[c] * c0) * std(i)
            specular AC, channel c     ~  I0 * u_s[c] * std(s)

        We use the channel where each component is *largest* as the reference
        amplitude. This is the conservative, "worst-case" matching that the
        prompt asks for.
        """
        ratio = float(self.noise_to_pulse_ratio)
        pulse_std = self.pulse_amplitude / np.sqrt(2.0)  # std of unit-amp sine

        pulse_ac_max = self.I0 * np.max(self.u_p) * pulse_std
        intensity_per_unit = self.I0 * np.max(self.u_c * self.c0)
        specular_per_unit = self.I0 * np.max(self.u_s)

        std_i = ratio * pulse_ac_max / intensity_per_unit
        std_s = ratio * pulse_ac_max / specular_per_unit
        return std_i, std_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> dict:
        """Build the full synthetic trace and return all components.

        Returns
        -------
        dict with keys:
            't'         : (N,) time vector [s]
            'C'         : (N, 3) noisy RGB trace
            'p'         : (N,) ground-truth pulse signal
            'i'         : (N,) intensity fluctuation
            's'         : (N,) specular fluctuation
            'v_n'       : (N, 3) sensor noise
            'fs'        : float
            'pulse_hz'  : float
            'u_c','u_s','u_p' : color vectors used
        """
        std_i, std_s = self._calibrate_amplitudes()

        p = self._pulse_signal()
        i = std_i * self._bandlimited_noise(0.05, 0.5)
        s = std_s * self._bandlimited_noise(0.1, 0.8)
        v_n = self.sensor_noise_std * self._rng.standard_normal((self.N, 3))

        # Reshape to broadcast: time on axis 0, RGB on axis 1.
        skin_dc = (self.u_c * self.c0)[None, :]                  # (1, 3)
        intensity_envelope = (1.0 + i)[:, None]                  # (N, 1)
        specular_term = self.u_s[None, :] * s[:, None]           # (N, 3)
        pulse_term = self.u_p[None, :] * p[:, None]              # (N, 3)

        # Eq. (6): C(t) = I0 * (1 + i(t)) * (u_c c_0 + u_s s(t) + u_p p(t)) + v_n
        C = self.I0 * intensity_envelope * (skin_dc + specular_term + pulse_term)
        C = C + v_n

        return {
            "t": self.t,
            "C": C,
            "p": p,
            "i": i,
            "s": s,
            "v_n": v_n,
            "fs": self.fs,
            "pulse_hz": self.pulse_bpm / 60.0,
            "u_c": self.u_c.copy(),
            "u_s": self.u_s.copy(),
            "u_p": self.u_p.copy(),
        }
