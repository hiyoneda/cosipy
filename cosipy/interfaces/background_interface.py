from typing import Protocol, runtime_checkable, Dict, Any
import histpy
import numpy as np

import logging
logger = logging.getLogger(__name__)

from .expectation_interface import BinnedExpectationInterface, UnbinnedExpectationInterface
from .measurements import Measurements

__all__ = [
           "BackgroundInterface",
           "BinnedBackgroundInterface",
           "UnbinnedBackgroundInterface",
           "NullBackground",
           ]

@runtime_checkable
class BackgroundInterface(Protocol):
    def set_parameters(self, **params:Dict[str, Any]) -> None:...
    @property
    def parameters(self) -> Dict[str, Any]:...

class BinnedBackgroundInterface(BackgroundInterface, BinnedExpectationInterface):
    """
    No new methods, just the inherited one
    """

class UnbinnedBackgroundInterface(BackgroundInterface, UnbinnedExpectationInterface):
    """
    No new methods, just the inherited one
    """

# Null background singleton
# It has not parameters and it always returns 0s
# It can be checked like a None. e.g.
# if bkg is not NullBackground:
#     expectation += bkg.get_expectation()
# Which should work even without the if, but the if
# allows to avoid a potentially (is it?) lenghty operation.
class _NullBackground(BinnedBackgroundInterface, UnbinnedBackgroundInterface):
    # All ways to instantiate this class should return the same object
    # The singleton instant will be define later, following the class
    # definition
    def __new__(cls, *args, **kwargs):
        return NullBackground
    def __copy__(self):
        return NullBackground
    def __deepcopy__(self, memo):
        return NullBackground
    def __call__(self):
        # This allows to use either NullBackground or NullBackground()
        # NullBackground is NullBackground() == True
        return NullBackground
    # Implement all method from binned and unbinned background
    # The results are all )'s
    @property
    def parameters(self): return {}
    def set_parameters(self, **params:Dict[str, Any]) -> None: pass
    def set_binning(self, axes:histpy.Axes) -> None: pass
    def get_expectation(self)->histpy.Histogram: pass
    def get_ncounts(self): return 0.
    def get_probability(self, measurements:Measurements): return np.broadcast_to(0., measurements.size)

# Instantiate *the* NullBackground singleton
try:
    NullBackground
except NameError:
    NullBackground = object.__new__(_NullBackground)
