from cosipy.interfaces import (BinnedLikelihoodInterface,
                               UnbinnedLikelihoodInterface,
                               BinnedDataInterface,
                               BinnedExpectationInterface,
                               BinnedBackgroundInterface,
                               )

import numpy as np

__all__ = ['UnbinnedLikelihood',
           'PoissonLikelihood']

class UnbinnedLikelihood(UnbinnedLikelihoodInterface):
    ...

class PoissonLikelihood(BinnedLikelihoodInterface):
    def __init__(self,
                 data: BinnedDataInterface,
                 response: BinnedExpectationInterface,
                 bkg: BinnedBackgroundInterface,
                 *args, **kwargs):

        self._data = data
        self._bkg = bkg
        self._response = response

    @property
    def has_bkg(self):
        return self._bkg is not None

    def get_log_like(self) -> float:

        # Compute expectation including background
        # If we don't have background, we won't modify the expectation, so
        # it's safe to use the internal cache.
        expectation = self._response.expectation(self._data.data.axes, copy = self.has_bkg)

        if self.has_bkg:
            # We won't modify the bkg expectation, so it's safe to use the internal cache
            expectation += self._bkg.expectation(self._data.data.axes, copy = False)

        # Get the arrays
        expectation = expectation.contents
        data = self._data.data.contents

        # Compute the log-likelihood:
        log_like = np.nansum(data * np.log(expectation) - expectation)

        return log_like

    @property
    def nobservations(self) -> int:
        return self._data.data.contents.size


