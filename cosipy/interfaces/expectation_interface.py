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
    def set_binning(self, axes:histpy.Axes) -> None:...
    def get_expectation(self)->histpy.Histogram:...

@runtime_checkable
class UnbinnedExpectationInterface(Protocol):
    def get_ncounts(self) -> float:...
    def get_probability(self, measurements:Measurements) -> np.ndarray:...



