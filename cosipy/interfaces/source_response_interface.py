from typing import Protocol, runtime_checkable
from astromodels import Model
from astromodels.sources import Source

from .expectation_interface import BinnedExpectationInterface, UnbinnedExpectationInterface

__all__ = ["SourceResponseInterface",
           "ThreeMLModelResponseInterface",
           "UnbinnedThreeMLModelResponseInterface",
           "BinnedThreeMLModelResponseInterface",
           "ThreeMLSourceResponseInterface",
           "UnbinnedThreeMLSourceResponseInterface",
           "BinnedThreeMLSourceResponseInterface"]

@runtime_checkable
class SourceResponseInterface(Protocol):
    ...
    def set_spacecraft_history(self, sc_history):

@runtime_checkable
class ThreeMLModelResponseInterface(SourceResponseInterface, Protocol):
    def set_model(self, model: Model):
        """
        The model is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """

@runtime_checkable
class UnbinnedThreeMLModelResponseInterface(UnbinnedExpectationInterface, ThreeMLModelResponseInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """

@runtime_checkable
class BinnedThreeMLModelResponseInterface(ThreeMLModelResponseInterface, BinnedExpectationInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """

@runtime_checkable
class ThreeMLSourceResponseInterface(SourceResponseInterface, Protocol):
    def set_source(self, source: Source):
        """
        The source is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
    def copy(self) -> "ThreeMLSourceResponseInterface":
        """
        This method is used to re-use the same object for multiple
        sources.
        It is expected to return a copy of itself, but deepcopy it's
        and any other necessary information such that when
        a new source is set, the expectation calculation
        are independent.

        psr1 = ThreeMLSourceResponse()
        psr2 = psr.copy()
        psr1.set_source(source1)
        psr2.set_source(source2)
        """

@runtime_checkable
class UnbinnedThreeMLSourceResponseInterface(UnbinnedExpectationInterface, ThreeMLSourceResponseInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """

@runtime_checkable
class BinnedThreeMLSourceResponseInterface(ThreeMLSourceResponseInterface, BinnedExpectationInterface, Protocol):
    """
    No new methods. Just the inherited ones.
    """


