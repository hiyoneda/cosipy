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

class UnbinnedDataInterface(DataInterface):
    @property
    def measurements(self) -> Measurements:...
    @property
    def measurement_types(self) -> Dict[str, Type[Any]]:...

class BinnedDataInterface(DataInterface):
    @property
    def axes(self) -> histpy.Axes:...
    @property
    def data(self) -> histpy.Histogram:...




