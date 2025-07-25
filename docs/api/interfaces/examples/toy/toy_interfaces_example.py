from typing import Dict, Any

from astromodels.sources import Source
from astromodels import LinearPolarization, SpectralComponent, Parameter
from astromodels.core.polarization import Polarization
import astropy.units as u
from cosipy import SpacecraftHistory
from cosipy.interfaces.background_interface import BackgroundDensityInterface
from cosipy.interfaces.data_interface import EventData, EventDataInterface

from cosipy.statistics import PoissonLikelihood, UnbinnedLikelihood


from cosipy.interfaces import (BinnedDataInterface,
                               BinnedBackgroundInterface,
                               BinnedThreeMLModelFoldingInterface,
                               BinnedThreeMLSourceResponseInterface,
                               ThreeMLPluginInterface, BackgroundInterface, FloatingMeasurement,
                               UnbinnedThreeMLSourceResponseInterface, UnbinnedThreeMLModelFoldingInterface)
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

class ToyData(BinnedDataInterface, EventData):
    # Random data. Normal signal on top of uniform bkg
    # Since the interfaces are Protocols, they don't *have*
    # to derive from the base class, but doing some helps
    # code readability, especially if you use an IDE.

    def __init__(self):
        self._data = Histogram(toy_axis)

        # Signal
        event_data = norm.rvs(size=nevents_signal)

        # Bkg
        bkg_event_data = uniform.rvs(toy_axis.lo_lim, toy_axis.hi_lim-toy_axis.lo_lim, size=nevents_bkg)

        # Join
        event_data = np.append(event_data, bkg_event_data)

        # Binned
        self._data.fill(event_data)

        #Unbinned
        measurements = FloatingMeasurement(event_data, 'x')
        EventData.__init__(self, measurements)

    @property
    def data(self) -> Histogram:
        return self._data

    @property
    def axes(self) -> Axes:
        return self._data.axes

class ToyBkg(BinnedBackgroundInterface, BackgroundDensityInterface):
    """
    Models a uniform background
    """

    def __init__(self):
        self._unit_expectation = Histogram(toy_axis)
        self._unit_expectation[:] = 1 / self._unit_expectation.nbins
        self._norm = 1

        # Doesn't need to be normalized
        self._unit_expectation_density = np.broadcast_to(1/(toy_axis.hi_lim - toy_axis.lo_lim), data.nevents)

    def set_parameters(self, **parameters:u.Quantity) -> None:
        self._norm = parameters['norm'].value

    def ncounts(self) -> float:
        return self._norm

    def expectation_density(self, data: EventDataInterface, copy: bool = True) -> np.ndarray:
        #Always a copy
        return self._norm*self._unit_expectation_density

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {'norm': u.Quantity(self._norm)}

    def expectation(self, data: BinnedDataInterface, copy = True) -> Histogram:

        if not isinstance(data, ToyData):
            raise TypeError(f"Wrong data type '{type(data)}', expected {ToyData}.")

        if data.axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        # Always a copy
        return self._unit_expectation * self._norm

class ToyPointSourceResponse(BinnedThreeMLSourceResponseInterface, UnbinnedThreeMLSourceResponseInterface):
    """
    This models a Gaussian signal in 1D, centered at 0 and with std = 1.
    The normalization --the "flux"-- is the only free parameters
    """

    def __init__(self):
        self._source = None
        self._unit_expectation = Histogram(toy_axis,
                                           contents=np.diff(norm.cdf(toy_axis.edges)))

    def ncounts(self) -> float:

        if self._source is None:
            raise RuntimeError("Set a source first")

        # Get the latest values of the flux
        # Remember that _model can be modified externally between calls.
        ns_events = self._source.spectrum.main.shape.k.value
        return ns_events

    def expectation_density(self, data:EventDataInterface, copy:bool) -> np.ndarray:

        if not isinstance(data, ToyData):
            raise TypeError(f"This class only support data of type {ToyData}")

        # I expect in the real case it'll be more efficient to compute
        # (ncounts, ncounts*prob) than (ncounts, prob)

        # Always copies
        return self.ncounts()*norm.pdf(data['x'].data)

    def set_source(self, source: Source):

        if not isinstance(source, PointSource):
            raise TypeError("I only know how to handle point sources!")

        self._source = source

    def expectation(self, data: BinnedDataInterface, copy = True) -> Histogram:

        if not isinstance(data, ToyData):
            raise TypeError(f"Wrong data type '{type(data)}', expected {ToyData}.")

        if data.axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        if self._source is None:
            raise RuntimeError("Set a source first")

        # Get the latest values of the flux
        # Remember that _model can be modified externally between calls.
        ns_events = self._source.spectrum.main.shape.k.value

        # Always copies
        return self._unit_expectation * ns_events

    def copy(self) -> "ToyPointSourceResponse":
        # We are not caching any results, so it's safe to do shallow copy without
        # re-initializing any member.
        return copy.copy(self)

class ToyModelFolding(BinnedThreeMLModelFoldingInterface, UnbinnedThreeMLModelFoldingInterface):

    def __init__(self, psr: BinnedThreeMLSourceResponseInterface):

        if not isinstance(psr, ToyPointSourceResponse):
            raise TypeError(f"Wrong psr type '{type(psr)}', expected {ToyPointSourceResponse}.")

        self._psr = psr
        self._psr_copies = {}

    def ncounts(self) -> float:

        ncounts = 0

        for source_name,psr in self._psr_copies.items():
            ncounts += psr.ncounts()

        return ncounts

    def expectation_density(self, data: EventDataInterface, copy:bool = True) -> np.ndarray:

        if not isinstance(data, ToyData):
            raise TypeError(f"This class only support data of type {ToyData}")

        expectation = np.zeros(data.nevents)

        for source_name, psr in self._psr_copies.items():
            expectation += psr.expectation_density(data, copy=False)

        # Always a copy
        return expectation

    def set_model(self, model: Model):

        self._psr_copies = {}
        for name,source in model.sources.items():
            psr_copy = self._psr.copy()
            psr_copy.set_source(source)
            self._psr_copies[name] = psr_copy

    def expectation(self, data: BinnedDataInterface, copy = True) -> Histogram:

        if not isinstance(data, ToyData):
            raise TypeError(f"Wrong data type '{type(data)}', expected {ToyData}.")

        expectation = Histogram(data.axes)

        for source_name,psr in self._psr_copies.items():
            expectation += psr.expectation(data, copy = False)

        # Always a copy
        return expectation

# ======= Actual code. This is how the "tutorial" will look like ================

# Set the inputs. These will eventually open file or set specific parameters,
# but since we are generating the data and models on the fly, and most parameter
# are hardcoded above withing the classes, then it's not necessary here.
data = ToyData()
psr = ToyPointSourceResponse()
response = ToyModelFolding(psr)
bkg = ToyBkg()

## Source model
## We'll just use the K value in u.cm / u.cm / u.s / u.keV
spectrum = Constant()
spectrum.k.value = 1

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
# Uncomment one. Either one works
#like_fun = PoissonLikelihood()
like_fun = UnbinnedLikelihood()

like_fun.set_data(data)
like_fun.set_response(response)
like_fun.set_background(bkg)
cosi = ThreeMLPluginInterface('cosi', like_fun)
plugins = DataList(cosi)
like = JointLikelihood(model, plugins)

# Before the fit, you can set the parameters initial values, bounds, etc.
# This is passed to the minimizer.
# In addition to model. Nuisance.
cosi.bkg_parameter['norm'].value = 1

# Run minimizer
like.fit()
print(like.minimizer)

# Plot results
fig, ax = plt.subplots()
data.data.plot(ax)
expectation = response.expectation(data)
if bkg is not None:
    expectation = expectation + bkg.expectation(data)
expectation.plot(ax)
plt.show()

# Grid
loglike = Histogram([np.linspace(.9*nevents_signal, 1.1*nevents_signal, 30), np.linspace(.9*nevents_bkg, 1.1*nevents_bkg, 31)], labels = ['s', 'b'])

for i,s in enumerate(loglike.axes['s'].centers):
    for j,b in enumerate(loglike.axes['b'].centers):

        spectrum.k.value = s
        cosi.bkg_parameter['norm'].value = b
        cosi._update_bkg_parameters() # Fix the need for this line

        loglike[i,j] = cosi.get_log_like()

loglike.plot()

plt.show()