from typing import Protocol, runtime_checkable

__all__ = ['LikelihoodInterface',
           'BinnedLikelihoodInterface',
           'UnbinnedLikelihoodInterface']

from .expectation_interface import UnbinnedExpectationInterface, BinnedExpectationInterface, ExpectationInterface
from .data_interface import UnbinnedDataInterface, BinnedDataInterface, DataInterface
from .background_interface import UnbinnedBackgroundInterface, BinnedBackgroundInterface, BackgroundInterface, \
    ThreeMLBackgroundInterface

@runtime_checkable
class LikelihoodInterface(Protocol):
    def __init__(self,
                 data: DataInterface,
                 response: ExpectationInterface,
                 bkg: BackgroundInterface,
                 *args, **kwargs):...
    def get_log_like(self) -> float:...
    @property
    def nobservations(self) -> int:
        """For BIC and other statistics"""
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
    def __init__(self,
                 data: BinnedDataInterface,
                 response: BinnedExpectationInterface,
                 bkg: BinnedBackgroundInterface,
                 *args, **kwargs):...

@runtime_checkable
class UnbinnedLikelihoodInterface(LikelihoodInterface, Protocol):
    """
        Needs to check that data, response and bkg are compatible
    """
    def __init__(self,
                 data: UnbinnedDataInterface,
                 response: UnbinnedExpectationInterface,
                 bkg: UnbinnedBackgroundInterface,
                 *args, **kwargs):...

