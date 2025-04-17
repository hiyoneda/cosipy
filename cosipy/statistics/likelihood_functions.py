from cosipy.interfaces import (BinnedLikelihoodInterface,
                               UnbinnedLikelihoodInterface,
                               BinnedDataInterface,
                               BinnedExpectationInterface,
                               BinnedBackgroundInterface,
                               NullBackground)

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

    def get_log_like(self) -> float:

        # Compute expectation including background
        expectation = self._response.expectation(self._data.data.axes)

        if self._bkg is not NullBackground:
            expectation = expectation + self._bkg.expectation(self._data.data.axes)

        # Get the arrays
        expectation = expectation.contents
        data = self._data.data.contents

        # Compute the log-likelihood:
        log_like = np.nansum(data * np.log(expectation) - expectation)

        return log_like

    @property
    def nobservations(self) -> int:
        return self._data.data.contents.size


