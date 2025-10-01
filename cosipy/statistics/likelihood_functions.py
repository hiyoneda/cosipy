import itertools
import logging
import operator

from cosipy import UnBinnedData
from cosipy.interfaces.expectation_interface import ExpectationInterface, ExpectationDensityInterface

logger = logging.getLogger(__name__)

from cosipy.interfaces import (BinnedLikelihoodInterface,
                               UnbinnedLikelihoodInterface,
                               BinnedDataInterface,
                               BinnedExpectationInterface,
                               BinnedBackgroundInterface, DataInterface, BackgroundInterface, EventDataInterface,
                               BackgroundDensityInterface,
                               )

import numpy as np

__all__ = ['UnbinnedLikelihood',
           'PoissonLikelihood']

class UnbinnedLikelihood(UnbinnedLikelihoodInterface):
    def __init__(self, data:EventDataInterface, response:ExpectationDensityInterface, bkg:BackgroundDensityInterface = None):

        self._data = data
        self._bkg = bkg
        self._response = response

    @property
    def data (self) -> EventDataInterface: return self._data
    @property
    def response(self) -> ExpectationDensityInterface: return self._response
    @property
    def bkg (self) -> BackgroundDensityInterface: return self._bkg

    @property
    def has_bkg(self):
        return self._bkg is not None

    @property
    def nobservations(self) -> int:
        return self._data.nevents

    def get_log_like(self) -> float:

        # Compute expectation including background

        ntot = self._response.ncounts()

        if self.has_bkg:

            ntot += self._bkg.ncounts()

            # Prevent 2 iteration over data using tee()
            data_iter_1, data_iter_2 = itertools.tee(self._data, 2)

            signal_density = self._response.expectation_density(data_iter_1)
            bkg_density = self._bkg.expectation_density(data_iter_2)

            density = np.fromiter(map(operator.add, signal_density, bkg_density), dtype=float)

        else:
            density = np.fromiter(self._response.expectation_density(), dtype=float)

        log_like = np.sum(np.log(density)) - ntot

        return log_like


class PoissonLikelihood(BinnedLikelihoodInterface):
    def __init__(self, data:BinnedDataInterface, response:BinnedExpectationInterface, bkg:BinnedBackgroundInterface = None):

        self._data = data
        self._bkg = bkg
        self._response = response

    @property
    def data (self) -> BinnedDataInterface: return self._data
    @property
    def response(self) -> BinnedExpectationInterface: return self._response
    @property
    def bkg (self) -> BinnedBackgroundInterface: return self._bkg

    @property
    def has_bkg(self):
        return self._bkg is not None

    @property
    def nobservations(self) -> int:
        return self._data.data.contents.size

    def get_log_like(self) -> float:

        # Compute expectation including background
        # If we don't have background, we won't modify the expectation, so
        # it's safe to use the internal cache.
        expectation = self._response.expectation(self._data.axes, copy = self.has_bkg)

        if self.has_bkg:
            # We won't modify the bkg expectation, so it's safe to use the internal cache
            expectation += self._bkg.expectation(self._data.axes, copy = False)

        # Get the arrays
        expectation = expectation.contents
        data = self._data.data.contents

        # Compute the log-likelihood:
        log_like = np.nansum(data * np.log(expectation) - expectation)

        return log_like

