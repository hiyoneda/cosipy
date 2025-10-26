from typing import Iterable, Tuple

from astropy.coordinates import Angle
from astropy.units import Quantity
from scipy.stats import rv_continuous, truncnorm, norm, uniform
from scipy.stats.sampling import SimpleRatioUniforms
import astropy.units as u
import numpy as np

from cosipy.interfaces.event import EmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonInterface
from cosipy.response.photon_types import PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface as PolDirESCPhoton
from scipy.special import erfi, erf

def _to_rad(angle):
    if isinstance(angle, (Quantity, Angle)):
        return angle.to_value(u.rad)
    else:
        return angle

class _RVSMixin:
    """
    Helper mixin for custom distributions (rv_continuous subclasses)
    that will likely only get a sample per setup

    Subclasses need to define _pdf and _cdf
    """

    def _rvs(self, *args, size=None, random_state=None):

        # Faster than default _rvs for large sizes, but slow setup
        # Most of the time we'll need a new setup per energy

        if size == tuple():
            # Weird default by rv_continous
            size = None

        if size is None:
            return super()._rvs(*args, size=size, random_state=random_state)
        else:

            rng = SimpleRatioUniforms(self, random_state=random_state)

            return rng.rvs(size=size)

class KleinNishinaPolarScatteringAngleDist(rv_continuous, _RVSMixin):
    """
    Klein-Nishina scattering angle distribution
    """

    def __init__(self, energy, *args, **kwargs):

        super().__init__(0, *args, a=0, b=np.pi, **kwargs)

        self._eps = energy.to_value(u.keV) / 510.99895069  # E/m_ec^2

        # Normalization
        # Mathematica
        # Integrate[(
        #  Sin[\[Theta]] (1 + 1/(
        #     1 + \[Epsilon] (1 - Cos[\[Theta]])) + \[Epsilon] (1 -
        #        Cos[\[Theta]]) -
        #     Sin[\[Theta]]^2))/(1 + \[Epsilon] (1 -
        #       Cos[\[Theta]]))^2, {\[Theta], 0, \[Pi]},
        #  Assumptions -> {\[Epsilon] > 0}]

        A = 2 * self._eps * (2 + self._eps * (1 + self._eps) * (8 + self._eps)) / (1 + 2 * self._eps) ** 2
        B = (-2 + self._eps * (self._eps - 2)) * np.log(1 + 2 * self._eps)

        self._norm = (A + B) / self._eps ** 3

    def _pdf(self, phi, *args):

        # Substitute Compton kinematic equation in Klein-Nishina dsigma/dOmega
        # Mathematica
        # eratio = 1 + self._eps (1 - cos_phi]) (*e/ep*)
        # (1/eratio)^2 (1/eratio + eratio - Sin[\[Theta]]^2) Sin[\[Theta]]

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        A = 1 + (1 / (1 + self._eps * (1 - cos_phi))) + self._eps * (1 - cos_phi) - sin_phi ** 2
        B = (1 + self._eps * (1 - cos_phi)) ** 2

        # Extra sin(phi) to account for phasespace
        return sin_phi * A / B / self._norm

    def _cdf(self, phi, *args):

        # Mathematica
        # Integrate[(
        #  Sin[\[Theta]] (1 + 1/(
        #     1 + self._eps (1 - cos_phi])) + self._eps (1 -
        #        cos_phi]) -
        #     Sin[\[Theta]]^2))/ ((1 + self._eps (1 -
        #       cos_phi]))^2), {\[Theta], 0, \[Theta]p},
        #  Assumptions -> {self._eps > 0, \[Theta]p < \[Pi], \[Theta]p  > 0}]

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        eps = self._eps
        eps2 = eps * eps
        eps3 = eps2 * eps

        A = 1 + eps - eps * cos_phi
        logA = np.log(A)
        B = (eps * (4 + 10 * eps + 8 * eps2 + eps3) \
             - 2 * eps3 * cos_phi ** 3 \
             + 2 * (1 + eps) ** 2 * (-2 - 2 * eps + eps2) * logA \
             + eps2 * cos_phi ** 2 * (6 + 10 * eps + eps2 + 2 * (-2 - 2 * eps + eps2) * logA) \
             - 2 * eps * cos_phi * (2 + 8 * eps + 8 * eps2 + eps3 \
                                    + 2 * (-2 - 4 * eps - eps2 + eps3) * logA))
        C = 2 * eps3 * A * A

        return B / C / self._norm

class KleinNishinaAzimuthalScatteringAngleDist(rv_continuous, _RVSMixin):

    def __init__(self, energy, theta, *args, **kwargs):
        """
        Conditional probability, given a polar angle and energy.

        Parameters
        ----------
        energy
        theta: polar angle
        args
        kwargs
        """

        super().__init__(0, *args, a=0, b=2*np.pi, **kwargs)

        theta = _to_rad(theta)

        # precompute some stuff
        self._eps = energy.to_value(u.keV) / 510.99895069  # E/m_ec^2
        self._sin_theta2 = np.sin(theta) ** 2
        self._energy_ratio =  1 + self._eps * (1 - np.cos(theta)) # From kinematics
        self._energy_ratio2 = self._energy_ratio * self._energy_ratio
        self._energy_ratio_inv = 1/self._energy_ratio
        self._energy_ratio_inv2 = self._energy_ratio_inv * self._energy_ratio_inv
        self._energy_ratio_inv3 = self._energy_ratio_inv2 * self._energy_ratio_inv

        # Mathematica
        # Integrate[(1/eratio + eratio - 2 sintheta2 Cos[\[Phi]]^2)/
        #   eratio^2, {\[Phi], 0, 2 \[Pi]},
        #   Assumptions -> {\[Epsilon] > 0}] // FullSimplify
        self._norm = 2 * np.pi * (1 + self._energy_ratio2 - self._sin_theta2 * self._energy_ratio) * self._energy_ratio_inv3

    def _pdf(self, phi, *args):
        """

        Parameters
        ----------
        phi: azimuthal angle, starting from the electric field vector direction
        args

        Returns
        -------

        """

        phi = _to_rad(phi)

        cos_phi = np.cos(phi)

        return (self._energy_ratio + self._energy_ratio_inv - 2 * self._sin_theta2 * cos_phi * cos_phi) * self._energy_ratio_inv2 / self._norm

    def _cdf(self, phi, *args):

        phi = _to_rad(phi)

        # Mathematica
        # Integrate[(1/eratio + eratio - 2 sintheta2 Cos[\[Phi]]^2)/
        #   eratio^2, {\[Phi], 0, \[Phi]lim},
        #   Assumptions -> {\[Epsilon] > 0}] // FullSimplify

        A = phi + phi*self._energy_ratio2 - self._energy_ratio * self._sin_theta2 * phi - self._energy_ratio * self._sin_theta2 * np.cos(phi) * np.sin(phi)

        return A * self._energy_ratio_inv3 / self._norm

class ARMNormDist(rv_continuous):

    def __init__(self, phi, angres, *args, **kwargs):
        """
        This accounts for the truncating effect since ARM is  limited to [-phi, pi-phi].
        It also accounts for the sin(phi+arm) phasespace

        Parameters
        ----------
        phi: Polar scattering angle
        angres: Standard deviation of the equivalent gaussian
        args
        kwargs
        """

        phi = _to_rad(phi)
        angres = _to_rad(angres)

        super().__init__(0, *args, a=-phi, b= np.pi - phi, **kwargs)

        # normalized such that int_0^pi random_arm = 1  (already includes sin(phi+arm))
        # Integrate[PDF[TruncatedDistribution[{0,\[Pi]},NormalDistribution [\[Phi],\[Sigma]]], x]Sin[x],{x,0,\[Pi]}]//Re//FullSimplify
        # Mathematica couldn't get only the real part analytically

        self._phi = phi
        self._angres = angres

        self._norm = np.real(
            np.exp(-(angres ** 2 / 2) - 1j * phi) *
            (1j * erf((np.pi + 1j * angres ** 2 - phi) / (np.sqrt(2) * angres)) +
             np.exp(2j * phi) * (erfi((angres ** 2 - 1j * phi) / (np.sqrt(2) * angres)) -
                                 erfi((1j * np.pi + angres ** 2 - 1j * phi) / (np.sqrt(2) * angres))) +
             erfi((angres ** 2 + 1j * phi) / (np.sqrt(2) * angres)))
            / (2 * (erf(phi / (np.sqrt(2) * angres)) - erf((-np.pi + phi) / (np.sqrt(2) * angres)))))

    def _pdf(self, arm, *args):

        return truncnorm.pdf(arm, -self._phi / self._angres, (np.pi - self._phi) / self._angres, 0, self._angres) * np.sin(self._phi + arm) / self._norm

    def _rvs(self, *args, size=None, random_state=None):

        rng = SimpleRatioUniforms(self, random_state=random_state)

        return rng.rvs(size=size)

class ThresholdKleinNishinaPolarScatteringAngleDist(KleinNishinaPolarScatteringAngleDist):

    def __init__(self, energy, energy_threshold=None, *args, **kwargs):

        super().__init__(energy)

        if energy_threshold is None:
            self._min_phi = 0
        else:

            # Mathematica
            # Solve[e/(e - edepmax) == 1 + \[Epsilon] (1 - (-1)), edepmax]

            max_energy_deposited = 2 * energy * self._eps / (1 + 2 * self._eps)

            if energy_threshold > max_energy_deposited:
                raise ValueError(
                    f"Threshold ({energy_threshold}) is greater than the maximum possible deposited energy ({max_energy_deposited}). PDF cannot be normalized")

            # Mathematica
            # Solve[e/(e - ethresh) ==
            #   1 + \[Epsilon] (1 - Cos[\[Theta]]), \[Theta] ]

            energy_threshold = energy_threshold.to_value(energy.unit)
            energy = energy.value

            eps_ediff = self._eps * (energy - energy_threshold)

            self._min_phi = np.arccos((eps_ediff - energy_threshold) / eps_ediff)

        # Renormalize
        self._cdf_min_phi = super()._cdf(self._min_phi)
        self._norm_factor = 1 / (1 - self._cdf_min_phi)

    def _renormalize(self, phi, prob):

        if np.isscalar(phi):
            if phi < self._min_phi:
                prob = 0
        else:
            prob = np.asarray(prob)
            phi = np.asarray(phi)
            prob[phi < self._min_phi] = 0

        prob *= self._norm_factor

        return prob

    def _pdf(self, phi, *args):

        phi = _to_rad(phi)

        prob = super()._pdf(phi, *args)

        return self._renormalize(phi, prob)

    def _cdf(self, phi, *args):

        phi = _to_rad(phi)

        cum_prob = super()._cdf(phi, *args) - self._cdf_min_phi

        return self._renormalize(phi, cum_prob)

class MeasuredEnergyDist(rv_continuous):

    def __init__(self, energy, energy_res, phi, full_absorp_prob, *args, **kwargs):
        """
        This is a *conditional* probability. We will assume the uncertainty on the measured angle phi is 0
        (all the CDS errors will come from the ARM distribution)

        If it is fully absorbed, then the deposited energy equal the initial energy.

        If it escaped, then it will assume that the deposited energy  corresponds to the energy of the first hit,
        following the Compton equation

        The measured energy will be drawn from a normal distribution
        centered at the deposited energy and std equal to energy_deposited*energy_res

        The geometry was not taking into account for the backscatter criterion since it was too complicated.

        Inputs and outputs are values assumed to be in the same units as input energy.

        Parameters
        ----------
        energy: initial energy.
        energy_res: energy resolution as fraction of the initial energy
        phi: polar scattered angle
        full_absorp_prob: probability of landing in the photopeak
        args
        kwargs
        """

        super().__init__(0, *args, a=0, **kwargs)

        if energy_res < 0 or energy_res > 1:
            raise ValueError(f"energy_res must be between [0,1]. Got {energy_res}")

        if full_absorp_prob < 0 or full_absorp_prob > 1:
            raise ValueError(f"full_absorp_prob must be between [0,1]. Got {full_absorp_prob}")

        eps = (energy / u.Quantity(510.99895069, u.keV)).value
        energy = energy.value

        phi = _to_rad(phi)
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        energy_deposited = energy / (1 + eps * (1 - np.cos(phi)))

        self._full_prob = full_absorp_prob
        self._partial_prob = 1 - full_absorp_prob

        self._dist_full = norm(loc=energy, scale=energy*energy_res)
        self._dist_partial = norm(loc=energy_deposited, scale=energy_deposited * energy_res)

    def _pdf(self, measured_energy, *args):
        return self._full_prob * self._dist_full.pdf(measured_energy) + self._partial_prob * self._dist_partial.pdf(measured_energy)

    def _cdf(self, measured_energy, *args):
        return self._full_prob * self._dist_full.cdf(measured_energy) + self._partial_prob * self._dist_partial.cdf(measured_energy)

    def _rvs(self, *args, size=None, random_state=None):

        full_absorp = uniform.rvs(size=size, random_state = random_state)  < self._full_prob

        nfull = np.count_nonzero(full_absorp)
        npartial = full_absorp.size - nfull

        samples = np.empty(full_absorp.shape)

        samples[full_absorp] = self._dist_full.rvs(*args, size=nfull, random_state=random_state)
        samples[np.logical_not(full_absorp)] = self._dist_partial.rvs(*args, size=npartial, random_state=random_state)

        return samples

class IdealComptonInstrumentResponseFunction(FarFieldInstrumentResponseFunctionInterface):

    # The photon class and event class that the IRF implementation can handle
    photon_type = PolDirESCPhoton
    event_type = EmCDSEventInSCFrameInterface

    def event_probability(self, query: Iterable[Tuple[PolDirESCPhoton, EmCDSEventInSCFrameInterface]]) -> Iterable[float]:
        """
        Return the probability density of measuring a given event given a photon.

        The units of the output the inverse of the phase space of the class event_type data space.
        e.g. if the event measured energy in keV, the units of output of this function are implicitly 1/keV
        """

        # P(phi) * P(E_dep | phi, nulambda, Ei) * P(Em | E_dep) * P(psichi | )

    def random_events(self, photons: Iterable[PolDirESCPhoton]) -> Iterable[EmCDSEventInSCFrameInterface]:
        """
        Return a stream of random events, photon by photon.

        The number of output event might be less than the number if input photons,
        since some might not be detected
        """

    def effective_area_cm2(self, photons: Iterable[PolDirESCPhoton]) -> Iterable[float]:
        """

        """

    def event_probability(self, query: Iterable[Tuple[PolDirESCPhoton, EmCDSEventInSCFrameInterface]]) -> Iterable[float]:
        """

        Parameters
        ----------
        query

        Returns
        -------

        """




