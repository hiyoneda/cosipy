from typing import Dict, Any

from astromodels.sources import Source
from astromodels import LinearPolarization, SpectralComponent, Parameter
from astromodels.core.polarization import Polarization
import astropy.units as u

from cosipy.statistics import PoissonLikelihood

from cosipy.interfaces import (BinnedDataInterface,
                               BinnedBackgroundInterface,
                               BinnedThreeMLModelResponseInterface,
                               BinnedThreeMLSourceResponseInterface,
                               ThreeMLPluginInterface, BackgroundInterface)
from histpy import Axis, Axes, Histogram
import numpy as np
from scipy.stats import norm, uniform

from threeML import Constant, PointSource, Model, JointLikelihood, DataList

from matplotlib import pyplot as plt

import copy

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

    def set_parameters(self, **parameters:u.Quantity) -> None:
        self._norm = parameters['norm'].value

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {'norm': u.Quantity(self._norm)}

    def expectation(self, axes: Axes, copy = True) -> Histogram:

        if axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        # Always a copy
        return self._unit_expectation * self._norm

class ToyPointSourceResponse(BinnedThreeMLSourceResponseInterface):
    """
    This models a Gaussian signal in 1D, centered at 0 and with std = 1.
    The normalization --the "flux"-- is the only free parameters
    """

    def __init__(self):
        self._source = None
        self._unit_expectation = Histogram(toy_axis,
                                           contents=np.diff(norm.cdf(toy_axis.edges)))

    def set_source(self, source: Source):

        if not isinstance(source, PointSource):
            raise TypeError("I only know how to handle point sources!")

        self._source = source

    def expectation(self, axes: Axes, copy = True) -> Histogram:
        if axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        if self._source is None:
            raise RuntimeError("Set a source first")

        # Get the latest values of the flux
        # Remember that _model can be modified externally between calls.
        flux = self._source.spectrum.main.shape.k.value

        # Always copies
        return self._unit_expectation * flux

    def copy(self) -> "ToyPointSourceResponse":
        return copy.copy(self)

class ToyModelResponse(BinnedThreeMLModelResponseInterface):

    def __init__(self, psr: BinnedThreeMLSourceResponseInterface):
        self._psr = psr
        self._psr_copies = {}

    def set_model(self, model: Model):

        self._psr_copies = {}
        for name,source in model.sources.items():

            psr_copy = self._psr.copy()
            psr_copy.set_source(source)
            self._psr_copies[name] = psr_copy

    def expectation(self, axes: Axes, copy = True) -> Histogram:
        expectation = Histogram(axes)

        for source_name,psr in self._psr_copies.items():
            expectation += psr.expectation(axes, copy = False)

        # Always a copy
        return expectation

# ======= Actual code. This is how the "tutorial" will look like ================

# Set the inputs. These will eventually open file or set specific parameters,
# but since we are generating the data and models on the fly, and most parameter
# are hardcoded above withing the classes, then it's not necessary here.
data = ToyData()
psr = ToyPointSourceResponse()
response = ToyModelResponse(psr)
bkg = ToyBkg()

## Source model
## We'll just use the K value in u.cm / u.cm / u.s / u.keV
spectrum = Constant()

polarized = False

if polarized:
    polarization = LinearPolarization(10, 10)
    polarization.degree.value = 0.
    polarization.angle.value = 10

    spectral_component = SpectralComponent('arbitrary_spectrum_name', spectrum, polarization)
    source = PointSource('arbitrary_source_name', 0, 0, components=[spectral_component])
else:

    source = PointSource("arbitrary_source_name",
                         l=0, b=0,  # Doesn't matter
                         spectral_shape=spectrum)

model = Model(source)

# Optional: Perform a background-only or a null-background fit
#bkg = None # Uncomment for no bkg
#model = Model() # Uncomment for bkg-only hypothesis

# Fit
cosi = ThreeMLPluginInterface('cosi', PoissonLikelihood(data, response, bkg))
plugins = DataList(cosi)
like = JointLikelihood(model, plugins)

# Before the fit, you can set the parameters initial values, bounds, etc.
# This is passed to the minimizer
cosi.bkg_parameter['norm'].value = 1
spectrum.k.value = 1

# Run minimizer
like.fit()
print(like.minimizer)

# Plot results
fig, ax = plt.subplots()
data.data.plot(ax)
expectation = response.expectation(data.data.axes)
if bkg is not None:
    expectation = expectation + bkg.expectation(data.data.axes)
expectation.plot(ax)
plt.show()
