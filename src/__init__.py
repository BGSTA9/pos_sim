
""" Plane-Orthogonal-to-Skin (POS) rPPG Simulation Package ---

This is a simulation package for the Plane-Orthogonal-to-Skin (POS) algorithm, from 
"Algorithmic Principles of Remote-PPG", by Wang, W., den Brinker, A. C., Stuijk, S., 
& de Haan, G. (2017). IEEE Transactions on Biomedical Engineering

DOI: 10.1109/TBME.2016.2609282

"""

from .generator import SyntheticDataGenerator
from .processor import POSProcessor
from .analyzer import SignalAnalyzer

__all__ = ["SyntheticDataGenerator", "POSProcessor", "SignalAnalyzer"]
