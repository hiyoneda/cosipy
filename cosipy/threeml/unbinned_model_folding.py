import itertools
from typing import Optional, Iterable

from astromodels import Model, PointSource, ExtendedSource

from cosipy.interfaces import UnbinnedThreeMLModelFoldingInterface, UnbinnedThreeMLSourceResponseInterface
from cosipy.interfaces.data_interface import EventDataInSCFrameInterface
from cosipy.response.threeml_response import ThreeMLModelFoldingCacheSourceResponsesMixin


class UnbinnedThreeMLModelFolding(UnbinnedThreeMLModelFoldingInterface, ThreeMLModelFoldingCacheSourceResponsesMixin):

    def __init__(self,
                 data: EventDataInSCFrameInterface,
                 point_source_response = UnbinnedThreeMLSourceResponseInterface,
                 extended_source_response: UnbinnedThreeMLSourceResponseInterface = None):

        # Interface inputs
        self._model = None

        # Implementation inputs
        self._psr = point_source_response
        self._esr = extended_source_response

        # Cache
        # Each source has its own cache.
        # We could cache the sum of all sources, but I thought
        # it was not worth it for the typical use case. Usually
        # at least one source changes in between call
        self._cached_model_dict = None
        self._source_responses = {}

    def set_model(self, model: Model):
        """
        The model is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
        self._model = model

    def ncounts(self) -> float:
        """
        Total expected counts
        """

        self._cache_source_responses()

        return sum(s.ncounts() for s in self._source_responses.values())

    def expectation_density(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:
        """
        Return the expected number of counts density from the start-th event
        to the stop-th event.

        Parameters
        ----------
        start : None | int
            From beginning by default
        stop: None|int
            Until the end by default
        """

        self._cache_source_responses()

        sources_expectation_iter = itertools.product(*(s.expectation_density(start, stop) for s in self._source_responses.values()))

        return [sum(expectations) for expectations in sources_expectation_iter]
