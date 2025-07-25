from typing import Protocol, runtime_checkable, Dict, Type, Any, Tuple, Iterator, Union

from histpy import Histogram, Axes

from .measurements import Measurement

import histpy

__all__ = ["DataInterface",
           "EventDataInterface",
           "BinnedDataInterface"]

@runtime_checkable
class DataInterface(Protocol):
    """
    Not much...
    """

@runtime_checkable
class EventDataInterface(DataInterface, Protocol):

    def __getitem__(self, item) -> Union[Tuple, Measurement]:
        """
        If item is:
        - str: the value of specific measurement for all events
        - int: all measurements for an specific event
        """

    def __iter__(self) -> Iterator[Tuple]:...

    @property
    def nevents(self) -> int:...

    @property
    def labels(self) -> Tuple[str]:...

    @property
    def types(self) -> Tuple[type]:...

    @property
    def nvars(self) -> int:...

class EventData(EventDataInterface):
    """
    Generic event data from measurement
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
        self._events = data
        self._labels = tuple([d.label for d in data])
        self._types = tuple([type(d) for d in data])
        self._value_types = tuple([d.value_type for d in data])

    def __getitem__(self, item):

        if isinstance(item, str):
            return self._events[self._labels.index(item)]
        elif isinstance(item, int):
            return tuple([d[item] for d in self._events])
        else:
            raise TypeError("Index must be either a measurement label or an entry position.")

    def __iter__(self):
        return zip(self._events)

    @property
    def nevents(self):
        return self._nevents

    @property
    def labels(self):
        return self._labels

    @property
    def types(self):
        return self._types

    @property
    def value_types(self):
        return self._value_types

    @property
    def nvars(self) -> int:
        return len(self._events)

@runtime_checkable
class BinnedDataInterface(DataInterface, Protocol):
    @property
    def data(self) -> Histogram:...
    @property
    def axes(self) -> Axes:...



