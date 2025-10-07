import itertools
import logging
import operator

from cosipy import UnBinnedData
from cosipy.interfaces.expectation_interface import ExpectationInterface, ExpectationDensityInterface
from cosipy.util.iterables import itertools_batched

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
    def __init__(self, response:ExpectationDensityInterface,
                 bkg:BackgroundDensityInterface = None,
                 batch_size:int = 100000):
        """
        Will get the number of events from the response and bkg expectation_density iterators

        Parameters
        ----------
        response
        bkg
        """

        self._bkg = bkg
        self._response = response
        self._nobservations = None

        self._batch_size = batch_size

    @property
    def has_bkg(self):
        return self._bkg is not None

    @property
    def nobservations(self) -> int:
        """
        Calling get_log_like first is faster, since we don't need to loop though the
        events
        """

        if self._nobservations is None:
            self._nobservations = sum(1 for _ in self._get_density_iter())

        return self._nobservations

    def _get_density_iter(self):

        if self.has_bkg:

            signal_density = self._response.expectation_density()
            bkg_density = self._bkg.expectation_density()

            return map(operator.add, signal_density, bkg_density)

        else:

            return self._response.expectation_density()

    def get_log_like(self) -> float:

        # Compute expectation including background

        ntot = self._response.ncounts()

        if self.has_bkg:
            ntot += self._bkg.ncounts()

        # It's faster to compute all log values at once, but requires keeping them in memory
        # Doing it by chunk is a compromise. We might need to adjust the chunk_size
        # Based on the system
        nobservations = 0
        density_log_sum = 0

        for density_iter_chunk in itertools_batched(self._get_density_iter(), self._batch_size):

            density = np.fromiter(density_iter_chunk, dtype=float)
            density_log_sum += np.sum(np.log(density))
            nobservations += density.size

        self._nobservations = nobservations

        log_like = density_log_sum - ntot

        return log_like


class PoissonLikelihood(BinnedLikelihoodInterface):
    def __init__(self, data:BinnedDataInterface,
                 response:BinnedExpectationInterface,
                 bkg:BinnedBackgroundInterface = None):

        self._data = data
        self._bkg = bkg
        self._response = response

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

