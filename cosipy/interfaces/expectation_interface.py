from typing import Protocol, runtime_checkable, Dict, Any

import histpy
import numpy as np
from cosipy.interfaces import BinnedDataInterface, EventDataInterface

__all__ = [
    "ExpectationDensityInterface",
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
class ExpectationDensityInterface(ExpectationInterface, Protocol):
    def ncounts(self) -> float:...
    def expectation_density(self, data:EventDataInterface, copy:bool) -> np.ndarray:
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



