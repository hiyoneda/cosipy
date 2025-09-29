from typing import Protocol, runtime_checkable, Dict, Any, Generator, Iterable, Optional, Union, Iterator

import histpy
import numpy as np
from cosipy.interfaces import BinnedDataInterface, EventDataInterface, DataInterface, Event

__all__ = [
    "ExpectationDensityInterface",
           "BinnedExpectationInterface"
           ]

from cosipy.interfaces.event_data_processor_interface import EventDataProcessorInterface


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
class ExpectationDensityInterface(ExpectationInterface, EventDataProcessorInterface, Protocol):

    def ncounts(self) -> float:
        """
        Total expected counts
        """

    def expectation_density(self, data: Iterable[Event]) -> Iterable[float]:
        return self.process(data)

    def get_binned_expectation(self, *args, **kwargs):
        raise NotImplementedError


