import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from histpy import Histogram, Axes, Axis, HealpixAxis

class SpacecraftAttitudeMap:
    """
    A time-weighted list of attitudes assumed by the spacecraft.  The
    weights have been modified to include only the time during which
    the specified source (if any) was not occluded by the earth.

    Parameters
    ----------
    attitudes : Attitude array
      attitudes assumed by the spacecraft
    weights : float ndarray
      time-weightings of each attitude
    source : SkyCoord, optional
      source used to trim time weights based on earth occultation;
      if None, we assume no trimming was performed
    """

    def __init__(self,
                 attitudes,
                 weights,
                 source = None):

        self.source = source
        self.attitudes = attitudes
        self.weights = weights

    def size(self):
        return len(self.attitudes)

    def get_axes_map(self,
                     nside,
                     scheme = 'ring',
                     coordsys = 'galactic',
                     axes = ('x', 'y')):
        """
        Translate the stored Attitudes to axes and produce a
        SpacecraftAxisMap from any two of them.

        Parameters
        ----------
        nside : int
          Healpix nside for map
        scheme : str, optional
          scheme for Healpix axes of map (default 'ring')
        coordsys : str, optional
          coordsys for Healpix axes of map (default 'galactic')
        labels : 2-tuple of str
          axes to plot; must be two of 'x', 'y', and 'z'

        Returns
        -------
        SpacecraftAxisMap

        """

        """
        dir_axis = (HealpixAxis(nside = nside,
                                scheme = scheme,
                                coordsys = coordsys))
        angle_axis = Axis(np.linspace(0., 2*np.pi, num=nside*8+1), unit=u.rad)

        m = Histogram([dir_axis, angle_axis], sparse = True, unit=u.s)

        rot_vecs   = self.attitudes[:-1].as_rotvec()
        rot_angles = np.linalg.norm(rot_vecs, axis=-1)
        rot_dirs   = rot_vecs / rot_angles[:,None]

        # discretize rotvecs for input Attitudes

        rd = SkyCoord(rot_dirs[:,0], rot_dirs[:,1], rot_dirs[:,2],
                      representation_type='cartesian',
                      frame = 'galactic')

        m.fill(rd, rot_angles, weight=self.weights[:-1])
        return m

        """

        ax_map = SpacecraftAxisMap(nside, scheme, coordsys, axes)

        x,y,z = self.attitudes[:-1].as_axes()
        ax_dict = { 'x' : x, 'y' : y, 'z' : z }

        # Fill histogram from whichever two axes the user requested
        ax_map.fill(ax_dict[axes[0]],
                    ax_dict[axes[1]],
                    weight = self.weights[:-1])

        return ax_map


class SpacecraftAxisMap(Histogram):
    """
    A map of weights for pairs of orthogonal axes.  The map is
    discretized onto two HealpixAxis axes of specified parameters.
    Representation is as a sparse 4D (= 2 Healpix grid) Histogram.

    This class should only be used for display; use
    SpacecraftAttitudeMap, which does not bin the axes, for actual
    computation.

    Parameters
    ----------
    nside : int
        Healpix nside for map
    scheme : str, optional
        scheme for Healpix axes of map (default 'ring')
    coordsys : str, optional
        coordsys for Healpix axes of map (default 'galactic')
    labels : 2-tuple of str
        labels for axes

    """
    def __init__(self,
                 nside,
                 scheme = 'ring',
                 coordsys = 'galactic',
                 labels = ('x', 'y')):

        axes = Axes((HealpixAxis(nside = nside,
                                 scheme = scheme,
                                 coordsys = coordsys,
                                 label = labels[0]),
                     HealpixAxis(nside = nside,
                                 scheme = scheme,
                                 coordsys = coordsys,
                                 label = labels[1])),
                    copy_axes=False)

        super().__init__(axes, sparse = True, unit = u.s)
