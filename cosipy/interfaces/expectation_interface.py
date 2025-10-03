from typing import Protocol, runtime_checkable, Dict, Any, Generator, Iterable, Optional, Union, Iterator

import histpy
import numpy as np
from histpy import Axes

from cosipy.interfaces import BinnedDataInterface, EventDataInterface, DataInterface, EventInterface

__all__ = [
    "ExpectationDensityInterface",
           "BinnedExpectationInterface"
           ]


@runtime_checkable
class ExpectationInterface(Protocol):
    pass

@runtime_checkable
class BinnedExpectationInterface(ExpectationInterface, Protocol):
    def expectation(self, axes:Axes, copy: Optional[bool])->histpy.Histogram:
        """

        Parameters
        ----------
        axes:
            Axes to bin the expectation into
        copy:
            If True (default), it will return an array that the user if free to modify.
            Otherwise, it will result a reference, possible to the cache, that
            the user should not modify

        Returns
        -------

        """

@runtime_checkable
class ExpectationDensityInterface(ExpectationInterface, Protocol):

    def ncounts(self) -> float:
        """
        Total expected counts
        """

    def expectation_density(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:
        """
        Return the expected number of counts density from the start-th event
        to the stop-th event.

        Parameters
        ----------
        start : None | int
            From beginning by default
        stop: None|int
            Until the end by default
        """


