from typing import Protocol, runtime_checkable, Dict, Type, Any

from .measurements import Measurements

import histpy

__all__ = ["DataInterface",
           "UnbinnedDataInterface",
           "BinnedDataInterface"]

@runtime_checkable
class DataInterface(Protocol):
    """
    Not much...
    """

@runtime_checkable
class UnbinnedDataInterface(DataInterface, Protocol):
    @property
    def measurements(self) -> Measurements:...

@runtime_checkable
class BinnedDataInterface(DataInterface, Protocol):
    @property
    def data(self) -> histpy.Histogram:...




