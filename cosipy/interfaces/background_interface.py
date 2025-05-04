from typing import Protocol, runtime_checkable, Dict, Any, Union
import histpy
import numpy as np

import logging

import astropy.units as u

from astromodels import Parameter

logger = logging.getLogger(__name__)

from .expectation_interface import BinnedExpectationInterface, UnbinnedExpectationInterface

__all__ = [
           "BackgroundInterface",
           "BinnedBackgroundInterface",
           "UnbinnedBackgroundInterface",
           ]

@runtime_checkable
class BackgroundInterface(Protocol):
    def set_parameters(self, **parameters:Dict[str, u.Quantity]) -> None:...
    @property
    def parameters(self) -> Dict[str, u.Quantity]:...

@runtime_checkable
class BinnedBackgroundInterface(BackgroundInterface, BinnedExpectationInterface, Protocol):
    """
    No new methods, just the inherited one
    """

@runtime_checkable
class UnbinnedBackgroundInterface(BackgroundInterface, UnbinnedExpectationInterface, Protocol):
    """
    No new methods, just the inherited one
    """
