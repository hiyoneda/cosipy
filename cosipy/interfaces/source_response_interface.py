from typing import Protocol, runtime_checkable
from astromodels import Model

from .expectation_interface import BinnedExpectationInterface, UnbinnedExpectationInterface

__all__ = ["SourceResponseInterface",
           "ThreeMLSourceResponseInterface",
           "ThreeMLUnbinnedSourceResponseInterface",
           "ThreeMLBinnedSourceResponseInterface"]

@runtime_checkable
class SourceResponseInterface(Protocol):
    ...

@runtime_checkable
class ThreeMLSourceResponseInterface(SourceResponseInterface, Protocol):
    def set_model(self, model: Model):
        """
        The model is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """

@runtime_checkable
class ThreeMLUnbinnedSourceResponseInterface(UnbinnedExpectationInterface, ThreeMLSourceResponseInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """

@runtime_checkable
class ThreeMLBinnedSourceResponseInterface(ThreeMLSourceResponseInterface, BinnedExpectationInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """
