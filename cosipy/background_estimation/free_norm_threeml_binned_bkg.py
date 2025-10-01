from typing import Dict, Tuple, Union, Any

import numpy as np
from astromodels import Parameter
from histpy import Histogram
from histpy import Axes

from astropy import units as u

from cosipy.interfaces import BinnedBackgroundInterface, BinnedDataInterface, DataInterface

__all__ = ["FreeNormBinnedBackground"]

class FreeNormBinnedBackground(BinnedBackgroundInterface):
    """
    This must translate to/from regular parameters
    with arbitrary type from/to 3ML parameters

    Parameter names are "{label}_norm". Default to "bkg_norm" is there was a single
    unlabeled component
    """

    _default_label = 'bkg'

    def __init__(self, distribution:Union[Histogram, Dict[str, Histogram]]):

        if isinstance(distribution, Histogram):
            # Single component
            self._distributions = {self._default_label: distribution}
            self._norms = 1.
        else:
            # Multiple label components.
            self._distributions = distribution
            self._norms = {f"{l}_norm":1. for l in self.labels}

        # These will be densify anyway since _expectation is dense
        # And histpy doesn't yet handle this operation efficiently
        # See Histogram._inplace_operation_handle_sparse()
        # Do it once and for all
        for label, bkg in self._distributions.items():
            if bkg.is_sparse:
                self._distributions[label] = bkg.to_dense()

        if self.ncomponents == 0:
            raise ValueError("You need to input at least one components")

        self._axes = None
        for bkg in self._distributions.values():
            if self._axes is None:
                self._axes = bkg.axes
            else:
                if self._axes != bkg.axes:
                    raise ValueError("All background components mus have the same axes")

        # Cache
        self._expectation = None
        self._last_norm_values = None

    @property
    def _single_component(self):
        return not isinstance(self._norms, dict)

    @property
    def norm(self):

        if not self._single_component:
            raise RuntimeError("This property can only be used for single-component models")

        return self._norms

    @property
    def norms(self):
        if self._single_component:
            return {f"{self._default_label}_norm": self._norms}
        else:
            return self._norms.items()

    @property
    def ncomponents(self):
        return len(self._distributions)

    @property
    def meausured_axes(self):
        return self._axes

    @property
    def labels(self):
        return self._distributions.keys()

    def set_norm(self, norm: Union[float, Dict[str, float]]):

        if self._single_component:
            if isinstance(norm, dict):
                self._norms = norm[f'{self._default_label}_norm']
            else:
                self._norms = norm
        else:
            # Multiple
            if not isinstance(norm, dict):
                raise TypeError("This a multi-component background. Provide labeled norm values in a dictionary")

            for label,norm_i in norm.items():
                if label not in self._norms.keys():
                    raise ValueError(f"Norm {label} not in {self._norms.keys()}")

                self._norms[label] = norm_i

    def set_parameters(self, **parameters:Dict[str, u.Quantity]) -> None:
        """
        Same keys as background components
        """

        self.set_norm({l:p.value for l,p in parameters.items()})

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {l:u.Quantity(n) for l,n in self.norms.items()}

    def expectation(self, axes:Axes, copy:bool = True)->Histogram:
        """

        Parameters
        ----------
        axes
        copy:
            If True, it will return an array that the user if free to modify.
            Otherwise, it will result a reference, possible to the cache, that
            the user should not modify

        Returns
        -------

        """

        if axes != self.meausured_axes:
            raise ValueError("Requested axes do not match the background component axes")

        # Check if we can use the cache
        if self._expectation is None:
            # First call. Initialize
            self._expectation = Histogram(self.meausured_axes)

        elif self.norms == self._last_norm_values:
            # No changes. Use cache
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation

        else:
            # First call or norms have change. Recalculate
            self._expectation.clear()

        # Compute expectation
        for norm,bkg in zip(self.norms.values(), self._distributions.values()):
            self._expectation += bkg * norm

        # Cache. Regular copy is enough since norm values are float en not mutable
        self._last_norm_values = self.norms.copy()

        if copy:
            return self._expectation.copy()
        else:
            return self._expectation

