import logging

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
    def __init__(self):

        self._data = None
        self._bkg = None
        self._response = None

    def set_data(self, data: DataInterface):
        super().set_data(data) # Checks type
        self._data = data

    def set_response(self, response: ExpectationInterface):
        super().set_response(response)  # Checks type
        self._response = response

    def set_background(self, bkg: BackgroundInterface):
        super().set_background(bkg)  # Checks type
        self._bkg = bkg

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

        if self._data is None:
            raise RuntimeError("Set the data before calling this function.")

        return self._data.nevents

    def get_log_like(self) -> float:

        if self._data is None or self._response is None:
            raise RuntimeError("Set data and response before calling this function.")

        # Compute expectation including background

        ntot = self._response.ncounts()

        # If we don't have background, we won't modify the expectation, so
        # it's safe to use the internal cache.
        density = self._response.expectation_density(self._data, copy = self.has_bkg)

        if self.has_bkg:

            ntot += self._bkg.ncounts()

            # We won't modify the bkg expectation, so it's safe to use the internal cache
            density += self._bkg.expectation_density(self._data, copy = False)

        # Compute the log-likelihood:
        log_like = np.sum(np.log(density)) - ntot

        return log_like


class PoissonLikelihood(BinnedLikelihoodInterface):
    def __init__(self):

        self._data = None
        self._bkg = None
        self._response = None

    def set_data(self, data: DataInterface):
        super().set_data(data) # Checks type
        self._data = data

    def set_response(self, response: ExpectationInterface):
        super().set_response(response)  # Checks type
        self._response = response

    def set_background(self, bkg: BackgroundInterface):
        super().set_background(bkg)  # Checks type
        self._bkg = bkg

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
        if self._data is None:
            raise RuntimeError("Set the data before calling this function.")

        return self._data.data.contents.size

    def get_log_like(self) -> float:

        if self._data is None or self._response is None:
            raise RuntimeError("Set data and response before calling this function.")

        # Compute expectation including background
        # If we don't have background, we won't modify the expectation, so
        # it's safe to use the internal cache.
        expectation = self._response.expectation(self._data, copy = self.has_bkg)

        if self.has_bkg:
            # We won't modify the bkg expectation, so it's safe to use the internal cache
            expectation += self._bkg.expectation(self._data, copy = False)

        # Get the arrays
        expectation = expectation.contents
        data = self._data.data.contents

        # Compute the log-likelihood:
        log_like = np.nansum(data * np.log(expectation) - expectation)

        return log_like

