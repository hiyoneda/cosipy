import itertools
from typing import Dict, Any, Generator, Iterator, Iterable, Optional, Union, Type

from astromodels.sources import Source
from astromodels import LinearPolarization, SpectralComponent, Parameter
from astromodels.core.polarization import Polarization
import astropy.units as u
from cosipy import SpacecraftHistory
from cosipy.interfaces.background_interface import BackgroundDensityInterface
from cosipy.interfaces.data_interface import EventDataInterface, DataInterface

from cosipy.statistics import PoissonLikelihood, UnbinnedLikelihood


from cosipy.interfaces import (BinnedDataInterface,
                               BinnedBackgroundInterface,
                               BinnedThreeMLModelFoldingInterface,
                               BinnedThreeMLSourceResponseInterface,
                               ThreeMLPluginInterface,
                               UnbinnedThreeMLSourceResponseInterface, UnbinnedThreeMLModelFoldingInterface, Event,
                               ThreeMLSourceResponseInterface)
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
shows how generic the interfaces can be. 
"""

# ======== Create toy interfaces for this model ===========

# Simple 1D axes. Hardcoded.
toy_axis = Axis(np.linspace(-5, 5), label = 'x')
nevents_signal = 1000
nevents_bkg = 1000
nevents_tot = nevents_signal + nevents_bkg

class ToyEvent(Event):
    """
    Unit-less 1D data of a measurement called "x" (could be anything)
    """

    def __init__(self, x):
        self._x = x

    @property
    def x(self):
        return self._x

class ToyData(DataInterface):

    @property
    def event_type(self) -> Type[Event]:
        return ToyEvent

class ToyEventData(EventDataInterface, ToyData):
    # Random data. Normal signal on top of uniform bkg

    def __init__(self):

        rng = np.random.default_rng()

        self._events = np.append(rng.normal(size = nevents_signal), rng.uniform(toy_axis.lo_lim, toy_axis.hi_lim, size = nevents_bkg))

        np.random.shuffle(self._events)

        self._nevents = nevents_tot

    def __iter__(self) -> Iterator[ToyEvent]:
        return iter(ToyEvent(x) for x in self._events)

    @property
    def nevents(self) -> int:
        return self._nevents

    def get_binned_data(self, axes:Axes, *args, **kwargs) -> "ToyBinnedData":

        binned_data = Histogram(axes)
        binned_data.fill(self._events)

        return ToyBinnedData(binned_data)

class ToyBinnedData(BinnedDataInterface, ToyData):

    def __init__(self, data:Histogram):

        if data.ndim != 1:
            raise ValueError("ToyBinnedData only take a 1D histogram")

        if data.axis.label != 'x':
            raise ValueError("ToyBinnedData requires an axis labeled 'x'")

        self._data = data

    @property
    def data(self) -> Histogram:
        return self._data

    @property
    def axes(self) -> Axes:
        return self._data.axes

class ToyBkg(BinnedBackgroundInterface, BackgroundDensityInterface):
    """
    Models a uniform background

    # Since the interfaces are Protocols, they don't *have*
    # to derive from the base class, but doing some helps
    # code readability, especially if you use an IDE.
    """

    def __init__(self):
        self._unit_expectation = Histogram(toy_axis)
        self._unit_expectation[:] = 1 / self._unit_expectation.nbins
        self._norm = 1

        # Doesn't need to be normalized
        self._unit_expectation_density = 1/(toy_axis.hi_lim - toy_axis.lo_lim)

    def set_parameters(self, **parameters:u.Quantity) -> None:
        self._norm = parameters['norm'].value

    def ncounts(self) -> float:
        return self._norm

    def expectation_density(self, data: Optional[Iterable[ToyEvent]] = None) -> Iterable[float]:

        density = self._norm * self._unit_expectation_density

        for _ in data:
            yield density

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {'norm': u.Quantity(self._norm)}

    def expectation(self, axes:Axes, copy = True) -> Histogram:

        if axes != self._unit_expectation.axes:
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

    def expectation_density(self, data:Optional[Iterable[ToyEvent]] = None) -> Iterable[float]:

        # I expect in the real case it'll be more efficient to compute
        # (ncounts, ncounts*prob) than (ncounts, prob)

        cache = self.ncounts()*norm.pdf([event.x for event in data])

        for n in cache:
            yield n

    def set_source(self, source: Source):

        if not isinstance(source, PointSource):
            raise TypeError("I only know how to handle point sources!")

        self._source = source

    def expectation(self, axes:Axes, copy = True) -> Histogram:

        if axes != self._unit_expectation.axes:
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

    def __init__(self, psr: ToyPointSourceResponse):

        self._model = None

        self._psr = psr
        self._psr_copies = {}

    def ncounts(self) -> float:

        ncounts = 0

        for source_name,psr in self._psr_copies.items():
            ncounts += psr.ncounts()

        return ncounts

    def expectation_density(self, data: Optional[Iterable[ToyEvent]] = None) -> Iterable[float]:

        self._cache_psr_copies()

        # One by one in this example, but they can also be done in chunks (e.g. with itertools batched or islice)
        for expectations in zip(*[p.expectation_density(d) for p,d in zip(self._psr_copies.values(), itertools.tee(data))]):
            yield np.sum(expectations)

    def set_model(self, model: Model):

        self._model = None

    def _cache_psr_copies(self):

        new_psr_copies = {}

        for name,source in model.sources.items():

            if name in self._psr_copies:
                # Use cache
                new_psr_copies[name] = self._psr_copies[name]

            psr_copy = self._psr.copy()
            psr_copy.set_source(source)

            new_psr_copies[name] = psr_copy

        self._psr_copies = new_psr_copies

    def expectation(self, axes:Axes, copy = True) -> Histogram:

        self._cache_psr_copies()

        expectation = Histogram(axes)

        for source_name,psr in self._psr_copies.items():
            expectation += psr.expectation(Axes(toy_axis), copy = False)

        # Always a copy
        return expectation

# ======= Actual code. This is how the "tutorial" will look like ================

# Binned or unbinned
unbinned = False

# Set the inputs. These will eventually open file or set specific parameters,
# but since we are generating the data and models on the fly, and most parameter
# are hardcoded above withing the classes, then it's not necessary here.
event_data = ToyEventData()
binned_data = event_data.get_binned_data(Axes(toy_axis))
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
if unbinned:
    like_fun = UnbinnedLikelihood()
    like_fun.set_data(event_data)
else:
    like_fun = PoissonLikelihood()
    like_fun.set_data(binned_data)

like_fun.set_response(response)
like_fun.set_background(bkg)
cosi = ThreeMLPluginInterface('cosi', like_fun)

# Before the fit, you can set the parameters initial values, bounds, etc.
# This is passed to the minimizer.
# In addition to model. Nuisance.
cosi.bkg_parameter['norm'].value = 1

plugins = DataList(cosi)
like = JointLikelihood(model, plugins)

# Run minimizer
like.fit()
print(like.minimizer)

# Plot results
fig, ax = plt.subplots()
binned_data.data.plot(ax)
expectation = response.expectation(binned_data.axes)
if bkg is not None:
    expectation = expectation + bkg.expectation(binned_data.axes)
expectation.plot(ax)
plt.show()

# Grid
loglike = Histogram([np.linspace(.9*nevents_signal, 1.1*nevents_signal, 30), np.linspace(.9*nevents_bkg, 1.1*nevents_bkg, 31)], labels = ['s', 'b'])

for i,s in enumerate(loglike.axes['s'].centers):
    for j,b in enumerate(loglike.axes['b'].centers):

        spectrum.k.value = s
        cosi.bkg_parameter['norm'].value = b

        loglike[i,j] = cosi.get_log_like()

loglike.plot()

plt.show()