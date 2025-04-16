from cosipy.interfaces import (NullBackground,
                               BackgroundInterface,
                               BinnedBackgroundInterface,
                               UnbinnedBackgroundInterface
                               )

def test_null_background():
    null_1 = NullBackground()
    null_2 = NullBackground
    null_3 = NullBackground

    assert null_1 is null_2
    assert null_2 is null_3
    assert null_3 is null_1
    assert isinstance(null_1, BackgroundInterface)
    assert isinstance(null_2, BinnedBackgroundInterface)
    assert isinstance(null_3, UnbinnedBackgroundInterface)

    class RandomClass: pass

    assert not isinstance(null_1, RandomClass)
