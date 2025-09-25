import copy

from cosipy.interfaces import BinnedThreeMLModelFoldingInterface, BinnedThreeMLSourceResponseInterface, \
    BinnedDataInterface, DataInterface

from astromodels import Model
from astromodels.sources import PointSource, ExtendedSource

from histpy import Axes, Histogram

__all__ = ["BinnedThreeMLModelFolding"]

class BinnedThreeMLModelFolding(BinnedThreeMLModelFoldingInterface):

    def __init__(self,
                 point_source_response:BinnedThreeMLSourceResponseInterface = None,
                 extended_source_response: BinnedThreeMLSourceResponseInterface = None):
        """

        Parameters
        ----------
        point_source_response:
            Response for :class:`astromodels.sources.PointSource`s.
            It can be None is you don't plan to use it for point sources.
        extended_source_response
            Response for :class:`astromodels.sources.ExtendedSource`s
            It can be None is you don't plan to use it for extended sources.
        """

        # Interface inputs
        self._data = None
        self._model = None

        # Implementation inputs
        self._psr = point_source_response
        self._esr = extended_source_response

        # Cache
        # Prevent unnecessary calculations and new memory allocations
        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._cached_model_dict = None
        self._source_responses = {}
        self._expectation = None

    def set_data(self, data: BinnedDataInterface):

        if self._expectation is None or self._expectation.axes != data.axes:
            # Needs new memory allocation, and recompute everything
            self._expectation = Histogram(data.axes)

        self._data = data

    def set_model(self, model: Model):
        """
        You need to call set_data() first.

        Parameters
        ----------
        model

        Returns
        -------

        """

        self._model = model

    def _cache_source_responses(self):
        """
        Create a copy of the PSR and ESR for each source
        Returns
        -------

        """

        # This accounts for the possibility of some sources being added or
        # removed from the model.
        new_source_responses = {}

        for name,source in self._model.sources.items():

            if name in self._source_responses:
                # Used cache
                new_source_responses[name] = self._source_responses[name]
                continue

            if isinstance(source, PointSource):

                if self._psr is None:
                    raise RuntimeError("The model includes a point source but no point source response was provided")

                psr_copy = self._psr.copy()
                psr_copy.set_source(source)
                psr_copy.set_data(self._data)
                new_source_responses[name] = psr_copy
            elif isinstance(source, ExtendedSource):

                if self._esr is None:
                    raise RuntimeError("The model includes an extended source but no extended source response was provided")

                esr_copy = self._esr.copy()
                esr_copy.set_source(source)
                esr_copy.set_data(self._data)
                new_source_responses[name] = esr_copy
            else:
                raise RuntimeError(f"The model contains the source {name} "
                                   f"of type {type(source)}. I don't know "
                                   "how to handle it!")

        self._source_responses = new_source_responses

    def expectation(self, copy:bool = True)->Histogram:
        """

        Parameters
        ----------
        data
        copy

        Returns
        -------

        """

        if self._data is None or self._model is None:
            raise RuntimeError("Call set_data() and set_model() first")

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        current_model_dict = self._model.to_dict()

        # If nothing has changed in the model, we can use the cached expectation
        # as is.
        # If the model has changed but the axes haven't, we can at least reuse
        # it and prevent new memory allocation, we just need to zero it out

        # TODO: currently Model.__eq__ seems broken. It returns. True even
        #  if the internal parameters changed. Caching the expected value
        #  is not implemented. Remove the "False and" when fixed
        if self._cached_model_dict is not None and self._cached_model_dict == current_model_dict:
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation
        else:
            self._expectation.clear()

        # Create a copy of the PSR and ESR for each source
        self._cache_source_responses()

        # Convolve all sources with the response
        for source_name,psr in self._source_responses.items():
            self._expectation += psr.expectation()

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._cached_model_dict = current_model_dict

        if copy:
            return self._expectation.copy()
        else:
            return self._expectation