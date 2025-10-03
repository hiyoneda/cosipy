import itertools
from typing import Dict, Any, Generator, Iterator, Iterable, Optional, Union, Type

from astromodels.sources import Source
from astromodels import LinearPolarization, SpectralComponent, Parameter
from astromodels.core.polarization import Polarization
import astropy.units as u
from astropy.time import Time
from astropy.units import Quantity
from numpy.ma.core import logical_or

from cosipy import SpacecraftHistory
from cosipy.event_selection.time_selection import TimeSelector
from cosipy.interfaces.background_interface import BackgroundDensityInterface
from cosipy.interfaces.data_interface import EventDataInterface, DataInterface, TimeTagEventDataInterface
from cosipy.interfaces.event import EventMetadata
from cosipy.interfaces.event_selection import EventSelectorInterface

from cosipy.statistics import PoissonLikelihood, UnbinnedLikelihood


from cosipy.interfaces import (BinnedDataInterface,
                               BinnedBackgroundInterface,
                               BinnedThreeMLModelFoldingInterface,
                               BinnedThreeMLSourceResponseInterface,
                               ThreeMLPluginInterface,
                               UnbinnedThreeMLSourceResponseInterface, UnbinnedThreeMLModelFoldingInterface, EventInterface,
                               ThreeMLSourceResponseInterface, TimeTagEventInterface)
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

class ToyEvent(TimeTagEventInterface, EventInterface):
    """
    Unit-less 1D data of a measurement called "x" (could be anything)
    """

    def __init__(self, index:int, x:float, time:Time):
        self._id = index
        self._x = x
        self._jd1 = time.jd1
        self._jd2 = time.jd2
        self._metadata = EventMetadata()

    @property
    def id(self):
        return self._id

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @property
    def x(self):
        return self._x

    @property
    def jd1(self):
        return self._jd1

    @property
    def jd2(self):
        return self._jd2

class ToyData(DataInterface):

    event_type = ToyEvent

class ToyEventDataLoader(ToyData):
    # This simulates reading event from file
    # Check that they are not being read twice

    def __init__(self):
        rng = np.random.default_rng()

        self._x = np.append(rng.normal(size=nevents_signal),
                            rng.uniform(toy_axis.lo_lim, toy_axis.hi_lim, size=nevents_bkg))

        self._tstart = Time("2000-01-01T00:00:00")
        self._tstop = Time("2000-01-02T00:00:00")

        dt = np.random.uniform(size=nevents_tot)
        dt_sort = np.argsort(dt)
        self._x = self._x[dt_sort]
        dt = dt[dt_sort]

        self._timestamps = self._tstart + dt * u.day

    def __iter__(self) -> Iterator[ToyEvent]:
        print("Loading events!")
        for n,(x,t) in enumerate(zip(self._x, self._timestamps)):
            yield ToyEvent(n,x,t)

class ToyEventData(TimeTagEventDataInterface, ToyData):
    # Random data. Normal signal on top of uniform bkg

    def __init__(self, loader:ToyEventDataLoader, selector:EventSelectorInterface = None):

        self._loader = selector(loader)
        self._cached_iter = None
        self._nevents = None  # After selection

    def __iter__(self) -> Iterator[ToyEvent]:

        if self._cached_iter is None:
            # First call. Split. Keep one and return the other
            self._loader, self._cached_iter = itertools.tee(self._loader)
            return self._cached_iter
        else:
            # Following calls: tee the loader again
            self._loader, new_iter = itertools.tee(self._loader)
            return new_iter

    @property
    def nevents(self) -> int:
        if self._nevents is None:
            # Not cached yet
            self._nevents = sum(1 for _ in self)

        return self._nevents

    @property
    def x(self):
        return np.asarray([e.x for e in self])

    @property
    def jd1(self) -> Iterable[float]:
        return np.asarray([e.jd1 for e in self])

    @property
    def jd2(self) -> Iterable[float]:
        return np.asarray([e.jd2 for e in self])

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

    def __init__(self, data: ToyEventData, duration:Quantity):

        self._data = data
        self._unit_expectation = Histogram(toy_axis)
        self._unit_expectation[:] = 1 / self._unit_expectation.nbins
        self._norm = 1

        self._sel_fraction = (duration/(1*u.day)).to_value('')
        self._unit_expectation_density = self._sel_fraction/(toy_axis.hi_lim - toy_axis.lo_lim)

    def set_parameters(self, **parameters:u.Quantity) -> None:
        self._norm = parameters['norm'].value

    def ncounts(self) -> float:
        return self._norm * self._sel_fraction

    def expectation_density(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:

        density = self._norm * self._unit_expectation_density

        for _ in itertools.islice(self._data, start, stop):
            yield density

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {'norm': u.Quantity(self._norm)}

    def expectation(self, axes:Axes, copy = True) -> Histogram:

        if axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        # Always a copy
        return self._unit_expectation * self._norm * self._sel_fraction

class ToyPointSourceResponse(BinnedThreeMLSourceResponseInterface, UnbinnedThreeMLSourceResponseInterface):
    """
    This models a Gaussian signal in 1D, centered at 0 and with std = 1.
    The normalization --the "flux"-- is the only free parameters
    """

    def __init__(self, data: ToyEventData, duration:Quantity):
        self._data = data
        self._source = None
        self._sel_fraction = (duration/(1*u.day)).to_value('')
        self._unit_expectation = Histogram(toy_axis,
                                           contents= self._sel_fraction * np.diff(norm.cdf(toy_axis.edges)))

    def ncounts(self) -> float:

        if self._source is None:
            raise RuntimeError("Set a source first")

        # Get the latest values of the flux
        # Remember that _model can be modified externally between calls.
        ns_events = self._sel_fraction * self._source.spectrum.main.shape.k.value
        return ns_events

    def expectation_density(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:

        # I expect in the real case it'll be more efficient to compute
        # (ncounts, ncounts*prob) than (ncounts, prob)

        cache = self.ncounts()*norm.pdf([event.x for event in itertools.islice(self._data, start, stop)])

        for n in cache:
            yield n

        # Alternative version without cache (slower)
        # for event in itertools.islice(self._data, start, stop):
        #     yield self.ncounts()*norm.pdf(event.x)


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

    def __init__(self, data:ToyEventData, psr: ToyPointSourceResponse):

        self._data = data
        self._model = None

        self._psr = psr
        self._psr_copies = {}

    def ncounts(self) -> float:

        ncounts = 0

        for source_name,psr in self._psr_copies.items():
            ncounts += psr.ncounts()

        return ncounts

    def expectation_density(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:

        self._cache_psr_copies()

        for expectations in zip(*[p.expectation_density(start, stop) for p in self._psr_copies.values()]):
            yield np.sum(expectations)

    def set_model(self, model: Model):

        self._model = model

    def _cache_psr_copies(self):

        new_psr_copies = {}

        for name,source in self._model.sources.items():

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

def get_binned_data(event_data:ToyEventData, axis:Axis) -> ToyBinnedData:
    """
    Only bins x axis
    """

    binned_data = Histogram(axis)
    binned_data.fill(event_data.x)

    return ToyBinnedData(binned_data)

# ======= Actual code. This is how the "tutorial" will look like ================

def main():

    # Binned or unbinned
    unbinned = True
    plot = True

    # Set the inputs. These will eventually open file or set specific parameters,
    # but since we are generating the data and models on the fly, and most parameter
    # are hardcoded above withing the classes, then it's not necessary here.
    tstart = Time("2000-01-01T01:00:00")
    tstop = Time("2000-01-01T10:00:00")
    duration = tstop - tstart
    selector = TimeSelector(tstart = tstart, tstop = tstop)

    data_loader = ToyEventDataLoader()
    event_data = ToyEventData(data_loader, selector=selector)

    psr = ToyPointSourceResponse(data = event_data, duration = duration)
    response = ToyModelFolding(data = event_data, psr = psr)
    bkg = ToyBkg(data = event_data, duration = duration)

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

    binned_data = None
    if plot or not unbinned:
        binned_data = get_binned_data(event_data, toy_axis)

    # Fit
    if unbinned:
        like_fun = UnbinnedLikelihood(response, bkg)
    else:
        like_fun = PoissonLikelihood(binned_data, response, bkg)


    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response = response,
                                  bkg = bkg)

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
    if plot:

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

if __name__ == "__main__":

    import cProfile
    cProfile.run('main()', filename = "prof_toy.prof")
    exit()

    main()