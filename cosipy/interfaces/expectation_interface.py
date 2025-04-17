from typing import Protocol, runtime_checkable, Dict, Any

import histpy
import numpy as np

from .measurements import Measurements

__all__ = [
           "UnbinnedExpectationInterface",
           "BinnedExpectationInterface"
           ]

@runtime_checkable
class BinnedExpectationInterface(Protocol):
    def expectation(self, axes:histpy.Axes)->histpy.Histogram:...

@runtime_checkable
class UnbinnedExpectationInterface(Protocol):
    @property
    def ncounts(self) -> float:...
    def probability(self, measurements:Measurements) -> np.ndarray:...



