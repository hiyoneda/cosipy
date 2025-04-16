from typing import Protocol, runtime_checkable
import threeML

from .expectation_interface import BinnedExpectationInterface, UnbinnedExpectationInterface

__all__ = ["SourceResponseInterface",
           "ThreeMLSourceResponseInterface",
           "ThreeMLUnbinnedSourceResponseInterface",
           "ThreeMLBinnedSourceResponseInterface"]

@runtime_checkable
class SourceResponseInterface(Protocol):
    ...

class ThreeMLSourceResponseInterface(SourceResponseInterface):
    def set_model(self, model: threeML.Model): ...


class ThreeMLUnbinnedSourceResponseInterface(UnbinnedExpectationInterface, ThreeMLSourceResponseInterface):
    """
    No new methods. Just the inherited ones.
    """

class ThreeMLBinnedSourceResponseInterface(ThreeMLSourceResponseInterface, BinnedExpectationInterface):
    """
    No new methods. Just the inherited ones.
    """
