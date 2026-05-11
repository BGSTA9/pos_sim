"""POS rPPG simulation package.

Implements the Plane-Orthogonal-to-Skin (POS) algorithm from
Wang, den Brinker, Stuijk, de Haan (2017), "Algorithmic Principles of
Remote-PPG", IEEE Trans. Biomed. Eng. 64(7), 1479-1491, together with
a physics-based synthetic RGB generator and a frequency-domain analyzer.
"""

from .generator import SyntheticDataGenerator
from .processor import POSProcessor
from .analyzer import SignalAnalyzer

__all__ = ["SyntheticDataGenerator", "POSProcessor", "SignalAnalyzer"]
