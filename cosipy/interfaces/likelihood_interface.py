from typing import Protocol, runtime_checkable

__all__ = ['LikelihoodInterface',
           'BinnedLikelihoodInterface',
           'UnbinnedLikelihoodInterface']

from .expectation_interface import UnbinnedExpectationInterface, BinnedExpectationInterface
from .data_interface import UnbinnedDataInterface, BinnedDataInterface
from .background_interface import UnbinnedBackgroundInterface, BinnedBackgroundInterface

@runtime_checkable
class LikelihoodInterface(Protocol):
    def get_log_like(self) -> float:...
    @property
    def nobservations(self) -> int:
        """For BIC and other statistics"""

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

