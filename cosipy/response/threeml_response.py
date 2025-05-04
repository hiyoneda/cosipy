import copy

from cosipy.interfaces import BinnedThreeMLModelResponseInterface, BinnedThreeMLSourceResponseInterface

from astromodels import Model
from astromodels.sources import PointSource, ExtendedSource

from histpy import Axes, Histogram

__all__ = ["BinnedThreeMLResponse"]

class BinnedThreeMLResponse(BinnedThreeMLModelResponseInterface):

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
        self._psr = point_source_response
        self._esr = extended_source_response
        self._source_responses = {}

        # Cache
        # Prevent unnecessary calculations and new memory allocations
        self._expectation = None
        self._model = None

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_model_dict = None

    def set_model(self, model: Model):
        """


        Parameters
        ----------
        model

        Returns
        -------

        """

        if model is self._model:
            # No need to do anything here
            return

        self._source_responses = {}

        for name,source in model.sources.items():

            if isinstance(source, PointSource):

                if self._psr is None:
                    raise RuntimeError("The model includes a point source but no point source response was provided")

                psr_copy = self._psr.copy()
                psr_copy.set_source(source)
                self._source_responses[name] = psr_copy
            elif isinstance(source, ExtendedSource):

                if self._esr is None:
                    raise RuntimeError("The model includes an extended source but no extended source response was provided")

                esr_copy = self._esr.copy()
                esr_copy.set_source(source)
                self._source_responses[name] = esr_copy
            else:
                raise RuntimeError(f"The model contains the source {name} "
                                   f"of type {type(source)}. I don't know "
                                   "how to handle it!")

        self._model = model

    def expectation(self, axes:Axes, copy:bool = True)->Histogram:
        """

        Parameters
        ----------
        axes
        copy

        Returns
        -------

        """
        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        current_model_dict = self._model.to_dict()

        if self._expectation is None or self._expectation.axes != axes:
            # Needs new memory allocation, and recompute everything
            self._expectation = Histogram(axes)
        else:
            # If nothing has changed in the model, we can use the cached expectation
            # as is.
            # If the model has changed but the axes haven't, we can at least reuse
            # is and prevent new memory allocation, we just need to zero it out

            # TODO: currently Model.__eq__ seems broken. It returns. True even
            #  if the internal parameters changed. Caching the expected value
            #  is not implemented. Remove the "False and" when fixed
            if self._last_convolved_model_dict == current_model_dict:
                if copy:
                    return self._expectation.copy()
                else:
                    return self._expectation
            else:
                self._expectation.clear()

        # Convolve all sources with the response
        for source_name,psr in self._source_responses.items():
            self._expectation += psr.expectation(axes)

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_model_dict = current_model_dict

        if copy:
            return self._expectation.copy()
        else:
            return self._expectation