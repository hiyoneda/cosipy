from typing import Dict, Tuple, Union, Any

import numpy as np
from astromodels import Parameter
from histpy import Histogram
from histpy import Axes

from cosipy.interfaces import ThreeMLBinnedBackgroundInterface, BinnedBackgroundInterface

__all__ = ["FreeNormBinnedBackground",
           "FreeNormThreeMLBinnedBackground"]

class FreeNormBinnedBackground(BinnedBackgroundInterface):
    """
    This must translate to/from regular parameters
    with arbitrary type from/to 3ML parameters

    """

    def __init__(self, *args:Tuple[Histogram], **kwargs:Dict[str, Histogram]):

        self._components = {}

        for n,bkg in enumerate(args):
            self._components[self._standardized_label(n)] = bkg

        for label, bkg in kwargs.items():
            if label in self.labels:
                raise ValueError("Repeated bkg component label.")

            self._components[label] = bkg

        # These will be densify anyway since _expectation is dense
        # And histpy doesn't yet handle this operation efficiently
        # See Histogram._inplace_operation_handle_sparse()
        # Do it once and for all
        for label, bkg in self._components.items():
            if bkg.is_sparse:
                self._components[label] = bkg.to_dense()

        if self.ncomponents == 0:
            raise ValueError("You need to input at least one components")

        self._axes = None
        for bkg in self._components.values():
            if self._axes is None:
                self._axes = bkg.axes
            else:
                if self._axes != bkg.axes:
                    raise ValueError("All background components mus have the same axes")

        self._norms = {l:1 for l in self.labels}

        # Cache
        self._expectation = None
        self._last_norm_values = None

    def _standardized_label(self, label:Union[str, int]):
        if isinstance(label, str):
            return label
        else:
            return f"bkg{label}"

    @property
    def norm(self):

        if self.ncomponents != 1:
            raise RuntimeError("This property can only be used for single-component models")

        return next(iter(self._norms.values()))

    @property
    def norms(self):
        return self._norms.values()

    @property
    def ncomponents(self):
        return len(self._components)

    @property
    def meausured_axes(self):
        return self._axes

    @property
    def labels(self):
        return self._components.keys()

    def set_norm(self, *args, **kwargs):

        for n,norm in enumerate(args):
            self._set_norm(n, norm)

        for label, bkg in kwargs.items():
            self._set_norm(label, norm)

    def _set_norm(self, label, norm):

        label = self._standardized_label(label)

        if label in self.labels:
            raise RuntimeError(f"Component {label} doesn't exist")

        self._norms[label] = norm

    def set_parameters(self, **parameters:Any) -> None:
        """
        Same keys as background components
        """

        self.set_norm(**parameters)

    @property
    def parameters(self) -> Dict[str, Any]:

        return self._norms

    def expectation(self, axes:Axes, copy:bool)->Histogram:
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

        elif self._norms == self._last_norm_values:
            # No changes. Use cache
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation

        else:
            # First call or norms have change. Recalculate
            self._expectation.clear()

        # Compute expectation
        for label in self.labels:
            self._expectation += self._norms[label] * self._components[label]

        # Cache. Regular copy is enough since norm values are float en not mutable
        self._last_norm_values = self._norms.copy()

        if copy:
            return self._expectation.copy()
        else:
            return self._expectation

class FreeNormThreeMLBinnedBackground(FreeNormBinnedBackground, ThreeMLBinnedBackgroundInterface):

    def __init__(self, *args:Tuple[Histogram], **kwargs:Dict[str, Histogram]):

        super().__init__(*args, **kwargs)

        # 3ML "Parameter" keeps track of a few more things than
        # a "bare" parameter.
        self._threeml_parameters = {label:Parameter(label, norm) for label,norm in self._norms.items()}

    def set_threeml_parameters(self, **parameters: Parameter):
        self._threeml_parameters = parameters
        self.set_parameters(**{label:parameter.value for label,parameter in parameters.items()})

    @property
    def threeml_parameters(self) -> Dict[str, Parameter]:
        return self._threeml_parameters





