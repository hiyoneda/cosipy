from typing import Protocol, runtime_checkable, Dict, Any

import histpy
import numpy as np
from cosipy.interfaces import BinnedDataInterface, EventDataInterface

__all__ = [
           "UnbinnedExpectationInterface",
           "BinnedExpectationInterface"
           ]

@runtime_checkable
class ExpectationInterface(Protocol):...

@runtime_checkable
class BinnedExpectationInterface(ExpectationInterface, Protocol):
    def expectation(self, data:BinnedDataInterface, copy:bool)->histpy.Histogram:
        """

        Parameters
        ----------
        data
        copy:
            If True, it will return an array that the user if free to modify.
            Otherwise, it will result a reference, possible to the cache, that
            the user should not modify

        Returns
        -------

        """

@runtime_checkable
class UnbinnedExpectationInterface(ExpectationInterface, Protocol):
    @property
    def ncounts(self) -> float:...
    def probability(self, data:EventDataInterface) -> np.ndarray:...



