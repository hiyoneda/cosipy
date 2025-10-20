import itertools
from typing import Union, Iterable

import numpy as np
from astropy.time import Time

from cosipy.interfaces import TimeTagEventInterface, EventInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import itertools_batched

from .good_time_interval import GoodTimeInterval

class EventSelectorGTI(EventSelectorInterface):

    def __init__(self, gti:GoodTimeInterval, batch_size:int = 10000):
        """
        Assumes events are time-ordered

        Parameters
        ----------
        gti:
        batch_size:
        """
        self._gti = gti

        self._batch_size = batch_size

    def _select(self, event:TimeTagEventInterface) -> bool:
        # Single event
        return next(iter(self.select([event])))

    def select(self, events:Union[TimeTagEventInterface, Iterable[TimeTagEventInterface]]) -> Union[bool, Iterable[bool]]:

        if isinstance(events, EventInterface):
            # Single event
            return self._select(events)
        else:
            # Multiple

            # Working in chunks/batches.
            # This can optimized based on the system

            for chunk in itertools_batched(events, self._batch_size):

                jd1 = []
                jd2 = []

                for event in chunk:
                    jd1.append(event.jd1)
                    jd2.append(event.jd2)

                time = Time(jd1, jd2, format = 'jd')

                selected, gti_index = self._gti.is_in_gti(time)

                for sel in selected:
                    yield sel
