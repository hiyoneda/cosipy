from typing import Protocol, runtime_checkable, Dict, Any, Iterator, Sequence, Generator, Iterable, Union, Optional

from . import Event

@runtime_checkable
class EventSelectorInterface(Protocol):

    def select(self, data: Optional[Iterable[Event]]) -> Iterable[bool]:
        """
        Returns True to keep an event, False to filter it out.
        """
