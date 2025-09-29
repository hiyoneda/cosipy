from typing import Protocol, runtime_checkable, Dict, Any, Iterator, Sequence, Generator, Iterable, Union, Optional

from . import Event
from .event_data_processor_interface import EventDataProcessorInterface

@runtime_checkable
class EventSelectorInterface(EventDataProcessorInterface, Protocol):

    def select(self, data: Optional[Iterable[Event]]) -> Iterable[bool]:
        """
        Returns True to keep an event, False to filter it out.
        """
        return self.process(data)
