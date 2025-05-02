from typing import Protocol, runtime_checkable, Dict, Any
import histpy
import numpy as np

import logging

from astromodels import Parameter

logger = logging.getLogger(__name__)

from .expectation_interface import BinnedExpectationInterface, UnbinnedExpectationInterface

__all__ = [
           "BackgroundInterface",
           "BinnedBackgroundInterface",
           "UnbinnedBackgroundInterface",
           "ThreeMLBackgroundInterface",
           "ThreeMLBinnedBackgroundInterface",
           "ThreeMLUnbinnedBackgroundInterface",
           ]

@runtime_checkable
class BackgroundInterface(Protocol):
    def set_parameters(self, **parameters:Any) -> None:...
    @property
    def parameters(self) -> Dict[str, Any]:...

@runtime_checkable
class ThreeMLBackgroundInterface(BackgroundInterface, Protocol):
    """
    This must translate to/from regular parameters
    with arbitrary type from/to 3ML parameters
    """

    def set_threeml_parameters(self, **parameters: Parameter):
        """
        The Parameter objects are passed "as reference", and can change.
        Remember to call set_parameters() before computing the expetation
        """
    @property
    def threeml_parameters(self)->Dict[str, Parameter]:
        """
        Note than we need more information (e.g. bounds) than what you
        get from base parameters property
        """
        return {} # Silence warning

@runtime_checkable
class BinnedBackgroundInterface(BackgroundInterface, BinnedExpectationInterface, Protocol):
    """
    No new methods, just the inherited one
    """

@runtime_checkable
class ThreeMLBinnedBackgroundInterface(BinnedBackgroundInterface, ThreeMLBackgroundInterface, Protocol):
    """
    No new methods, just the inherited one
    """

@runtime_checkable
class UnbinnedBackgroundInterface(BackgroundInterface, UnbinnedExpectationInterface, Protocol):
    """
    No new methods, just the inherited one
    """

@runtime_checkable
class ThreeMLUnbinnedBackgroundInterface(BinnedBackgroundInterface, ThreeMLBackgroundInterface, Protocol):
    """
    No new methods, just the inherited one
    """
