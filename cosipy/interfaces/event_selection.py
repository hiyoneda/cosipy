from typing import Protocol, runtime_checkable, Dict, Any, Iterator, Sequence, Generator, Iterable, Union, Optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Guard to prevent circular import
    from .data_interface import EventDataInterface

@runtime_checkable
class EventWeightingInterface(Protocol):
    """
    3 calling mechanism

    1.
    weights.set_data(data)
    weights.weight()

    In this case weight() will call iter(data)

    2.
    weights.weight(data)

    In this case weight() will first call set_data(data) (if needed), and then iter(data).

    3.
    weights.set_data(data)
    weights.weight(iterator)

    This prevents weight() from calling iter(data). However, it is assumed that
    iterator is equivalent to iter(data). This allows to use cached versions
    of the iterator or itertools.tee.
    """

    def set_data(self, data:'EventDataInterface'):...

    def weight(self, data: Optional[Union['EventDataInterface', Iterator]]) -> Iterable[float]:...

@runtime_checkable
class EventSelectorInterface(EventWeightingInterface, Protocol):

    def set_data(self, data:'EventDataInterface'):...

    def select(self, data: Optional[Union['EventDataInterface', Iterator]]) -> Iterable[bool]:
        """
        Returns True to keep an event, False to filter it out.
        """

    def weight(self, data: Optional[Union['EventDataInterface', Iterator]]) -> Iterable[float]:
        return self.select(data)
