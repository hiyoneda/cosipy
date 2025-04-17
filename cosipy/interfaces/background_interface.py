from typing import Protocol, runtime_checkable, Dict, Any
import histpy
import numpy as np

import logging

from astromodels import Parameter

logger = logging.getLogger(__name__)

from .expectation_interface import BinnedExpectationInterface, UnbinnedExpectationInterface
from .measurements import Measurements

__all__ = [
           "BackgroundInterface",
           "BinnedBackgroundInterface",
           "UnbinnedBackgroundInterface",
           "ThreeMLBackgroundInterface",
           "ThreeMLBinnedBackgroundInterface",
           "ThreeMLUnbinnedBackgroundInterface",
           "NullBackground",
           ]

@runtime_checkable
class BackgroundInterface(Protocol):
    def set_parameters(self, **params:Dict[str, Any]) -> None:...
    @property
    def parameters(self) -> Dict[str, Any]:...

class ThreeMLBackgroundInterface(BackgroundInterface):
    """
    This must translate to/from regular parameters
    with arbitrary type from/to 3ML parameters

    """
    def set_threeml_parameters(self, **parameters: Dict[str, Parameter]):
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

class BinnedBackgroundInterface(BackgroundInterface, BinnedExpectationInterface):
    """
    No new methods, just the inherited one
    """

class ThreeMLBinnedBackgroundInterface(BinnedBackgroundInterface, ThreeMLBackgroundInterface):
    """
    No new methods, just the inherited one
    """

class UnbinnedBackgroundInterface(BackgroundInterface, UnbinnedExpectationInterface):
    """
    No new methods, just the inherited one
    """

class ThreeMLUnbinnedBackgroundInterface(BinnedBackgroundInterface, ThreeMLBackgroundInterface):
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
class _NullBackground(ThreeMLBinnedBackgroundInterface, ThreeMLUnbinnedBackgroundInterface):
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
    @property
    def threeml_parameters(self) ->Dict[str, Parameter]: return {}
    def set_parameters(self, **params:Dict[str, Any]) -> None: pass
    def set_threeml_parameters(self, **parameters: Dict[str, Parameter]): pass
    def expectation(self, axes:histpy.Axes)->histpy.Histogram: return histpy.Histogram(axes)
    @property
    def ncounts(self): return 0.
    @property
    def probability(self, measurements:Measurements): return np.broadcast_to(0., measurements.size)

# Instantiate *the* NullBackground singleton
try:
    NullBackground
except NameError:
    NullBackground = object.__new__(_NullBackground)
