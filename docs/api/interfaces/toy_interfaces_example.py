from typing import Dict, Any

from cosipy.threeml import COSILike
from cosipy.interfaces import (BinnedDataInterface,
                               BinnedBackgroundInterface,
                               ThreeMLBinnedBackgroundInterface,
                               ThreeMLBinnedSourceResponseInterface)
from histpy import Axis, Axes, Histogram
import numpy as np
from scipy.stats import norm, uniform

from threeML import Constant, PointSource, Model, JointLikelihood, DataList, Parameter

from matplotlib import pyplot as plt

"""
This is an example on how to use the new interfaces.

To keep things simple, example itself is a toy model.
It a 1D model, with a Gaussian signal on top of a flat
uniform background. You can execute it until the end
to see a plot on how it looks like.

It looks nothing like COSI data, but
shows how generic the interfaces can be. I'm still working
on refactoring our current code to this format.
"""


# ======== Create toy interfaces for this model ===========

# Simple 1D axes. Hardcoded.
toy_axis = Axis(np.linspace(-5, 5))
nevents_signal = 1000
nevents_bkg = 1000

class ToyData(BinnedDataInterface):
    # Random data. Normal signal on top of uniform bkg
    # Since the interfaces are Protocols, they don't *have*
    # to derive from the base class, but doing some helps
    # code readability, especially if you use an IDE.

    def __init__(self):
        self._data = Histogram(toy_axis)

        # Signal
        self._data.fill(norm.rvs(size=nevents_signal))

        # Bkg
        self._data.fill(uniform.rvs(toy_axis.lo_lim, toy_axis.hi_lim-toy_axis.lo_lim, size=nevents_bkg))

    @property
    def data(self) -> Histogram:
        return self._data


class ToyBkg(BinnedBackgroundInterface):
    """
    Models a uniform background
    """

    def __init__(self):
        self._unit_expectation = Histogram(toy_axis)
        self._unit_expectation[:] = 1 / self._unit_expectation.nbins
        self._norm = 1

    def set_parameters(self, **params: Dict[str, Any]) -> None:
        self._norm = params['norm']

    @property
    def parameters(self) -> Dict[str, Any]:
        return {'norm': self._norm}

    def expectation(self, axes: Axes) -> Histogram:

        if axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        return self._unit_expectation * self._norm


class ToyThreeMLBkg(ToyBkg, ThreeMLBinnedBackgroundInterface):
    """
    This class extends the core ToyBkg class by providing the extra
    "translation" methods needed to interface with 3ML.
    """

    def __init__(self):

        super().__init__()

        # 3ML "Parameter" keeps track of a few more things than
        # a "bare" parameter.
        self._threeml_parameters = {'norm':Parameter('norm', self._norm)}

    def expectation(self, axes: Axes) -> Histogram:
        # Overrides ToyBkg expectation
        # Update, inn case it changed externally
        self.set_parameters(norm = self._threeml_parameters['norm'].value)

        return super().expectation(axes)

    @property
    def threeml_parameters(self) -> Dict[str, Parameter]:
        return self._threeml_parameters

    def set_threeml_parameters(self, norm: Parameter, **kwargs):
        self._threeml_parameters['norm'] = norm
        self.set_parameters(norm = norm.value)


class ToySourceResponse(ThreeMLBinnedSourceResponseInterface):
    """
    This models a Gaussian signal in 1D, centered at 0 and with std = 1.
    The normalization --the "flux"-- is the only free parameters
    """

    def __init__(self):
        self._model = None
        self._unit_expectation = Histogram(toy_axis,
                                           contents=np.diff(norm.cdf(toy_axis.edges)))

    def set_model(self, model: Model):
        self._model = model

    def expectation(self, axes: Axes) -> Histogram:
        if axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        if self._model is None:
            raise RuntimeError("Set model first")

        # Get the latest values of the flux
        # Remember that _model can be modified externally between calls.
        sources = self._model.sources

        if len(sources) == 0:
            flux = 0.
        else:
            flux = self._model.sources['source'].spectrum.main.shape.k.value

        return self._unit_expectation * flux


# ======= Actual code. This is how the "tutorial" will look like ================

# Set the inputs. These will eventually open file or set specific parameters,
# but since we are generating the data and models on the fly, and most parameter
# are hardcoded above withing the classes, then it's not necessary here.
data = ToyData()
response = ToySourceResponse()
bkg = ToyThreeMLBkg()

## Source model
## We'll just use the K value in u.cm / u.cm / u.s / u.keV
spectrum = Constant()
source = PointSource("source",  # arbitrary, but needs to be unique
                     l=0, b=0,  # Doesn't matter
                     spectral_shape=spectrum)
model = Model(source)

# Here you can set the parameters initial values, bounds, etc.
# This is passed to the minimizer
bkg.threeml_parameters['norm'].value = 1
spectrum.k.value = 1

# Optional: Perform a background-only or a null-background fit
#bkg = None # Uncomment for no bkg
#model = Model() # Uncomment for bkg-only hypothesis

# Fit
cosi = COSILike('cosi', data, response, bkg)
plugins = DataList(cosi)
like = JointLikelihood(model, plugins)
like.fit()

# Plot results
fig, ax = plt.subplots()
data.data.plot(ax)
expectation = response.expectation(data.data.axes)
if bkg is not None:
    expectation = expectation + bkg.expectation(data.data.axes)
expectation.plot(ax)
plt.show()
