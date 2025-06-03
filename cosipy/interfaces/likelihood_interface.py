from typing import Protocol, runtime_checkable

__all__ = ['LikelihoodInterface',
           'BinnedLikelihoodInterface',
           'UnbinnedLikelihoodInterface']

from .expectation_interface import UnbinnedExpectationInterface, BinnedExpectationInterface, ExpectationInterface
from .data_interface import UnbinnedDataInterface, BinnedDataInterface, DataInterface
from .background_interface import UnbinnedBackgroundInterface, BinnedBackgroundInterface, BackgroundInterface

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
        if not isinstance(data, UnbinnedDataInterface):
            raise TypeError("Incorrect data type for unbinned likelihood.")

    def set_response(self, response: ExpectationInterface):
        if not isinstance(response, UnbinnedExpectationInterface):
            raise TypeError("Incorrect data type for unbinned likelihood.")

    def set_background(self, bkg: BackgroundInterface):
        if not isinstance(bkg, UnbinnedBackgroundInterface):
            raise TypeError("Incorrect background type for unbinned likelihood.")

