from typing import Protocol, Optional, Iterable

from cosipy.interfaces import EventDataInterface, Event

class EventDataProcessorInterface(Protocol):
    """
    Get a output per event

    Iterables can be anything. The implementations do not necessarily need to
    process the data event by event.
    """

    def set_data(self, data: EventDataInterface):...

    def process(self, data: Optional[Iterable[Event]]) -> Iterable:
        """
        2 calling mechanisms

        1.
        processor.process()

        In this case process() will call iter(data), where data was passed though set_data()

        2.
        processor.process(data_subset: Iterable[float])

        The implementation will use the general data properties from the set_data() call,
        but will call iter(data_subset) instead.
        This allows the user to cache event data, in addition to looping over only a
        portion of the data
        """