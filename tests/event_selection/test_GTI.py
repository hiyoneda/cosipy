import pytest
import numpy as np
from astropy.time import Time

from cosipy.event_selection import GoodTimeInterval

def test_GTI(tmp_path):

    tstarts = Time([60970.0, 60980.0], format='mjd', scale = 'utc')
    tstops  = Time([60975.0, 60985.0], format='mjd', scale = 'utc')

    gti = GoodTimeInterval(tstarts, tstops)

    assert len(gti) == 2

    tstarts = gti.tstart_list
    tstops  = gti.tstop_list

    for i, (tstart, tstop) in enumerate(gti):
        assert tstart == tstarts[i] == gti[i][0]
        assert tstop  == tstops[i] == gti[i][1]

    # save file
    gti.save_as_fits(tmp_path / 'gti.fits')
    gti_from_fits = GoodTimeInterval.from_fits(tmp_path / 'gti.fits')

    assert np.all(tstarts == gti_from_fits.tstart_list)
    assert np.all(tstops  == gti_from_fits.tstop_list)

    # intersection

    #GTI1
    tstarts_1 = Time([60970.0, 60980.0], format='mjd', scale = 'utc')
    tstops_1  = Time([60975.0, 60985.0], format='mjd', scale = 'utc')
    
    gti1 = GoodTimeInterval(tstarts_1, tstops_1)
    
    #GTI2
    tstarts_2 = Time([60972.0, 60979.0], format='mjd', scale = 'utc')
    tstops_2  = Time([60977.0, 60983.0], format='mjd', scale = 'utc')
    
    gti2 = GoodTimeInterval(tstarts_2, tstops_2)
    
    #GTI3
    tstarts_3 = Time([60970.0], format='mjd', scale = 'utc')
    tstops_3  = Time([60990.0], format='mjd', scale = 'utc')
    
    gti3 = GoodTimeInterval(tstarts_3, tstops_3)
    
    #Intersection
    gti_intersection = GoodTimeInterval.intersection(gti1, gti2, gti3)

    assert np.all(gti_intersection.tstart_list == Time([60972.0, 60980.0], format='mjd', scale = 'utc'))
    assert np.all(gti_intersection.tstop_list  == Time([60975.0, 60983.0], format='mjd', scale = 'utc'))
