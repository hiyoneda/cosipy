from histpy import Histogram#, Axes, Axis

import numpy as np
import astropy.units as u
from scoords import SpacecraftFrame, Attitude

from astromodels.core.polarization import Polarization, LinearPolarization, StokesPolarization

from .functions import get_integrated_spectral_model

import logging
logger = logging.getLogger(__name__)

class PointSourceResponse(Histogram):
    """
    Handles the multi-dimensional matrix that describes the expected
    response of the instrument for a particular point in the sky.

    Parameters
    ----------
    axes : :py:class:`histpy.Axes`
        Binning information for each variable. The following labels are expected:\n
        - ``Ei``: Real energy
        - ``Em``: Measured energy. Optional
        - ``Phi``: Compton angle. Optional.
        - ``PsiChi``:  Location in the Compton Data Space (HEALPix pixel). Optional.
        - ``SigmaTau``: Electron recoil angle (HEALPix pixel). Optional.
        - ``Dist``: Distance from first interaction. Optional.
    contents : array, :py:class:`astropy.units.Quantity` or :py:class:`sparse.SparseArray`
        Array containing the differential effective area convolved with wht source exposure.
    unit : :py:class:`astropy.units.Unit`, optional
        Physical units, if not specified as part of ``contents``. Units of ``area*time``
        are expected.
    """
    
    @property
    def photon_energy_axis(self):
        """
        Real energy bins (``Ei``).

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        
        return self.axes['Ei']

    @property
    def measurement_axes(self):
        return self.axes['Em', 'Phi', 'PsiChi']

    def get_expectation(self, spectrum, polarization=None):
        """
        Convolve the response with a spectral (and optionally, polarization) hypothesis to obtain the expected
        excess counts from the source.

        Parameters
        ----------
        spectrum : :py:class:`threeML.Model`
            Spectral hypothesis.
        polarization : 'astromodels.core.polarization.LinearPolarization', optional
            Polarization angle and degree. The angle is assumed to have same convention as point source response.
        
        Returns
        -------
        :py:class:`histpy.Histogram`
             Histogram with the expected counts on each analysis bin
        """

        # FIXME: the logic of this code block should be moved to 3ML.
        #   We want to see if the source is polarized, and if so, confirm
        #   transform to linear polarization.
        #   https://github.com/threeML/astromodels/blob/master/astromodels/core/polarization.py
        if polarization is not None:

            if type(polarization) == Polarization:
                # FIXME: Polarization is the base class, but a 3ML source
                #   with no polarization default to the base class.
                #   The base class shouldn't be able to be instantiated,
                #   and we should have a NullPolarization subclass or None
                polarization = None

            elif isinstance(polarization, LinearPolarization):

                if polarization.degree.value is 0:
                    polarization = None

            elif isinstance(polarization, StokesPolarization):

                # FIXME: Here we should convert the any Stokes parameters to Linear
                #    The circular component looks like unpolarized to us.
                #    This conversion is not yet implemented in Astromodels
                raise ValueError("Fix me. I can't handle StokesPolarization yet")

            else:

                if isinstance(polarization, Polarization):
                    raise TypeError(f"Fix me. I don't know how to handle this polarization type")
                else:
                    raise TypeError(f"Polarization must be a Polarization subclass")


        if polarization is None:

            if 'Pol' in self.axes.labels:

                raise RuntimeError("Must include polarization in point source response if using polarization response")

            contents = self.contents

        else:

            if not 'Pol' in self.axes.labels:
                
                raise RuntimeError("Response must have polarization angle axis to include polarization in point source response")

            polarization_angle = polarization.angle.value
            polarization_level = polarization.degree.value / 100.

            if polarization_angle == 180.:
                polarization_angle = 0.

            unpolarized_weights = np.full(self.axes['Pol'].nbins, (1. - polarization_level) / self.axes['Pol'].nbins)
            polarized_weights = np.zeros(self.axes['Pol'].nbins)

            polarization_bin_index = self.axes['Pol'].find_bin(polarization_angle * u.deg)
            polarized_weights[polarization_bin_index] = polarization_level

            weights = unpolarized_weights + polarized_weights

            contents = np.tensordot(weights, self.contents, axes=([0], [self.axes.label_to_index('Pol')]))


        energy_axis = self.photon_energy_axis

        flux = get_integrated_spectral_model(spectrum, energy_axis)
        
        expectation = np.tensordot(contents, flux.contents, axes=([0], [0]))
        
        # Note: np.tensordot loses unit if we use a sparse matrix as it input.
        if self.is_sparse:
            expectation *= self.unit * flux.unit

        hist = Histogram(self.measurement_axes, contents=expectation)

        if not hist.unit == u.dimensionless_unscaled:
            raise RuntimeError("Expectation should be dimensionless, but has units of " + str(hist.unit) + ".")

        return hist
