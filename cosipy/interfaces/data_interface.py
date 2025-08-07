from typing import Protocol, runtime_checkable, Dict, Type, Any, Tuple, Iterator, Union

import numpy as np
from .event_selection import EventSelectorInterface
from histpy import Histogram, Axes

from .measurements import Measurement

import histpy

__all__ = ["DataInterface",
           "EventDataInterface",
           "BinnedDataInterface",
           "EventData"]

@runtime_checkable
class DataInterface(Protocol):
    """
    Not much...
    """

@runtime_checkable
class EventDataInterface(DataInterface, Protocol):

    def __getitem__(self, item:Union[str, int]) -> Union[Tuple, Measurement]:
        """
        If item is:
        - str: the value of specific measurement for all events
        - int: all measurements for an specific event (whether masked or unmasked)
        """

    def __iter__(self) -> Iterator[Tuple]:
        """
        Only loops through unmasked values
        """

    @property
    def nevents(self) -> int:...

    @property
    def labels(self) -> Tuple[str]:...

    @property
    def nmeasurements(self) -> int:
        """
        Number of Measurements. Each measurement can potentially have more tha one value
        --e.g. RA,Dec can be considered a single measurement
        """

    def set_selection(self, selection: Union[EventSelectorInterface, None]) -> None:
        """
        None would drop the selection. Implementation might not implement the ability to drop
        a selection when the underlying data was discarded for efficiency reasons.
        """

    @property
    def selection(self) -> Union[EventSelectorInterface, None]:
        """
        The current selection set
        """

    @property
    def nselected(self) -> int:...


class EventData(EventDataInterface):
    """
    Generic event data from measurement set
    """

    def __init__(self, *data:Measurement):

        # Check shame
        size = None

        for data_i in data:

            if size is None:
                size = data_i.size
            else:
                if size != data_i.size:
                    raise ValueError("All measurement arrays must have the same size")

        self._nevents = size
        self._nselected = size
        self._events = data
        self._labels = tuple([d.label for d in data])
        self._selection = None

    def __getitem__(self, item:[Union[str, int]]) -> Union[Tuple, Measurement]:

        if isinstance(item, str):
            return self._events[self._labels.index(item)]
        elif isinstance(item, int):
            return tuple([d[item] for d in self._events])
        else:
            raise TypeError("Index must be either a measurement label or an entry position.")

    def __iter__(self) -> Iterator[Tuple]:
        return zip(*self._events)

    @property
    def nevents(self):
        return self._nevents

    @property
    def labels(self):
        return self._labels

    @property
    def nmeasurements(self) -> int:
        return len(self._events)

    def set_selection(self, selection:EventSelectorInterface) -> None:

        if selection is None:
            self._selection = None
            self._nselected = self._nevents
        else:

            self._selection = selection

            # Signals the need to recompute this number
            self._nselected = -1

    @property
    def selection(self) -> EventSelectorInterface:
        return self._selection

    def nselected(self) -> int:

        if self._nselected == -1:
            # Not yet cached since last set selection
            self._nselected = sum(self._selection.select(self))

        return self._nselected



@runtime_checkable
class BinnedDataInterface(DataInterface, Protocol):
    @property
    def data(self) -> Histogram:...
    @property
    def axes(self) -> Axes:...



