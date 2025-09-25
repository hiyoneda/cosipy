from typing import Protocol, runtime_checkable, Dict, Any, Generator, Iterable, Optional, Union, Iterator

import histpy
import numpy as np
from cosipy.interfaces import BinnedDataInterface, EventDataInterface, DataInterface

__all__ = [
    "ExpectationDensityInterface",
           "BinnedExpectationInterface"
           ]

@runtime_checkable
class ExpectationInterface(Protocol):
    def set_data(self, data: DataInterface):...

@runtime_checkable
class BinnedExpectationInterface(ExpectationInterface, Protocol):
    def expectation(self, copy: Optional[bool])->histpy.Histogram:
        """

        Parameters
        ----------
        copy:
            If True (default), it will return an array that the user if free to modify.
            Otherwise, it will result a reference, possible to the cache, that
            the user should not modify

        Returns
        -------

        """

@runtime_checkable
class ExpectationDensityInterface(ExpectationInterface, Protocol):
    """
    3 calling mechanisms

    1.
    expectation.set_data(data)
    expectation.expectation_density()

    In this case expectation_density() will call iter(data)

    2.
    expectation.expectation_density(data)

    In this case expectation_density() will first call set_data(data) (if needed), and then iter(data).

    3.
    expectation.set_data(data)
    expectation.expectation_density(iterator)

    This prevents expectation_density() from calling iter(data). However, it is assumed that
    iterator is equivalent to iter(data). This allows to use cached versions
    of the iterator or itertools.tee.
    """

    def ncounts(self) -> float:...
    def expectation_density(self, data: Optional[Union['EventDataInterface', Iterator]]) -> Iterable[float]:
        """
        Parameters
        ----------
        data

        Returns
        -------

        """



