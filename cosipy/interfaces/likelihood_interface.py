from typing import Protocol, runtime_checkable

__all__ = ['LikelihoodInterface',
           'BinnedLikelihoodInterface',
           'UnbinnedLikelihoodInterface']

from .expectation_interface import ExpectationDensityInterface, BinnedExpectationInterface, ExpectationInterface
from .data_interface import BinnedDataInterface, DataInterface, EventDataInterface
from .background_interface import BackgroundDensityInterface, BinnedBackgroundInterface, BackgroundInterface

@runtime_checkable
class LikelihoodInterface(Protocol):
    def get_log_like(self) -> float:...
    @property
    def nobservations(self) -> int:
        """For BIC and other statistics"""
    def set_data(self, data: DataInterface):...
    def set_response(self, response: ExpectationInterface): ...
    def set_background(self, bkg: BackgroundInterface): ...
    @property
    def data (self) -> DataInterface: ...
    @property
    def response(self) -> ExpectationInterface: ...
    @property
    def bkg (self) -> BackgroundInterface: ...

@runtime_checkable
class BinnedLikelihoodInterface(LikelihoodInterface, Protocol):
    """
    Needs to check that data, response and bkg are compatible
    """
    def set_data(self, data: DataInterface):
        if not isinstance(data, BinnedDataInterface):
            raise TypeError("Incorrect data type for binned likelihood.")

    def set_response(self, response: ExpectationInterface):
        if not isinstance(response, BinnedExpectationInterface):
            raise TypeError("Incorrect data type for binned likelihood.")

    def set_background(self, bkg: BackgroundInterface):
        if not isinstance(bkg, BinnedBackgroundInterface):
            raise TypeError("Incorrect background type for binned likelihood.")

@runtime_checkable
class UnbinnedLikelihoodInterface(LikelihoodInterface, Protocol):
    """
        Needs to check that data, response and bkg are compatible
    """
    def set_data(self, data: DataInterface):
        if not isinstance(data, EventDataInterface):
            raise TypeError("Incorrect data type for unbinned likelihood.")

    def set_response(self, response: ExpectationInterface):
        if not isinstance(response, ExpectationDensityInterface):
            raise TypeError("Incorrect data type for unbinned likelihood.")

    def set_background(self, bkg: BackgroundInterface):
        if not isinstance(bkg, BackgroundDensityInterface):
            raise TypeError("Incorrect background type for unbinned likelihood.")

