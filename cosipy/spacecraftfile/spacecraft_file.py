from pathlib import Path

import numpy as np

import astropy.units as u

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, GCRS, ITRS
from docutils.io import Input
from histpy import Histogram, TimeAxis
from mhealpy import HealpixMap

from scoords import Attitude, SpacecraftFrame

from .scatt_map import SpacecraftAttitudeMap

from typing import Union

import logging
logger = logging.getLogger(__name__)

__all__ = ["SpacecraftFile"]

class SpacecraftFile:

    def __init__(self,
                 obstime: Time,
                 attitude: Attitude,
                 location: Union[EarthLocation, GCRS, ITRS],
                 livetime: u.Quantity = None):
        """
        Handles the spacecraft orientation. Calculates the dwell obstime
        map and point source response over a certain orientation period. 
        Exports the point source response as RMF and ARF files that can be read by XSPEC.
        
        Parameters
        ----------
        obstime:
            The obstime stamps for each pointings. Note this is NOT the obstime duration, see "livetime".
        attitude:
            Spacecraft orientation with respect to an inertial system.
        location:
            Location of the spacecraft at each timestamp.
        livetime:
            Time the instrument was live for the corresponding
            obstime bin. Should have one less element than the number of
            timestamps. If not provided, it will assume that the instrument
            was fully on without interrruptions.
        """

        time_axis = TimeAxis(obstime, copy = False, label= 'obstime')

        if livetime is None:
            livetime = time_axis.widths.to(u.s)

        self._hist = Histogram(time_axis, livetime, copy_contents = False)

        if not (location.shape == () or location.shape == obstime.shape):
            raise ValueError(f"'location' must be a scalar or have the same length as the timestamps ({obstime.shape}), but it has shape ({location.shape})")

        if not (attitude.shape == () or attitude.shape == obstime.shape):
            raise ValueError(f"'attitude' must be a scalar or have the same length as the timestamps ({obstime.shape}), but it has shape ({attitude.shape})")

        self._attitude = attitude

        self._location = self._standardize_location(location)

    def _standardize_location(self, location: Union[EarthLocation, GCRS, ITRS]):

        if isinstance(location, EarthLocation):
            # Already the standard format
            return location

        elif isinstance(location, GCRS):
            # GCRS -> ITRS and call again
            return self._standardize_location(location.transform_to(ITRS(self.obstime)))

        elif isinstance(location, ITRS):
            # ITRS -> EarthLocation
            return location.earth_location

        else:
            raise TypeError(f"Location type {type(location)} not supported.")

    @property
    def nintervals(self):
        return self._hist.nbins

    @property
    def intervals_duration(self):
        return self._hist.axis.widths.to(self._hist.unit)

    @property
    def intervals_tstart(self):
        return self._hist.axis.lower_bounds

    @property
    def intervals_tstop(self):
        return self._hist.axis.upper_bounds

    @property
    def tstart(self):
        return self._hist.axis.lo_lim

    @property
    def tstop(self):
        return self._hist.axis.hi_lim

    @property
    def npoints(self):
        return self._hist.nbins + 1

    @property
    def obstime(self):
        return self._hist.axis.edges

    @property
    def livetime(self):
        return self._hist.contents

    @property
    def attitude(self):
        return self._attitude

    @property
    def location(self)->EarthLocation:
        return self._location

    @classmethod
    def open(cls, file) -> "SpacecraftFile":

        """
        Parses timestamps, axis positions from file and returns to __init__.

        Parameters
        ----------
        file : str
            The file path of the pointings.

        Returns
        -------
        cosipy.spacecraftfile.spacecraft_file
            The SpacecraftFile object.
        """

        file = Path(file)

        if file.suffix == ".ori":
            return cls._parse_from_file(file)
        else:
            raise ValueError(f"File format for {file} not supported")

    @classmethod
    def _parse_from_file(cls, file) -> "SpacecraftFile":
        """
        Parses an .ori txt file with MEGAlib formatting.

        # Columns
        # 0: Always "OG" (orbital geometry)
        # 1: obstime: timestamp in unix seconds
        # 2: lat_x: galactic latitude of SC x-axis (deg)
        # 3: lon_x: galactic longitude of SC x-axis (deg)
        # 4: lat_z galactic latitude of SC z-axis (deg)
        # 5: lon_z: galactic longitude of SC y-axis (deg)
        # 6: altitude: altitude above from Earth's ellipsoid (km)
        # 7: Earth_lat: galactic latitude of the direction the Earth's zenith is pointing to at the SC location (deg)
        # 8: Earth_lon: galactic longitude of the direction the Earth's zenith is pointing to at the SC location (deg)
        # 9: livetime (previously called SAA): accumulated uptime up to the following entry (seconds)

        Parameters
        ----------
        file:
            Path to .ori file

        Returns
        -------
        cosipy.spacecraftfile.spacecraft_file
            The SpacecraftFile object.
        """

        time,lat_x,lon_x,lat_z,lon_z,altitude,earth_lat,earth_lon,livetime = orientation_file = np.loadtxt(file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), unpack = True,
                                                                                                      delimiter=' ', skiprows=1, comments=("#", "EN"))
        time = Time(time, format="unix")

        xpointings = SkyCoord(l=lon_x * u.deg, b=lat_x * u.deg, frame="galactic")
        zpointings = SkyCoord(l=lon_z * u.deg, b=lat_z * u.deg, frame="galactic")

        attitude = Attitude.from_axes(x=xpointings, z=zpointings, frame = 'galactic')

        livetime = livetime[:-1]*u.s # The last element is 0.

        # Currently, the orbit information is in a weird format.
        # The altitude it's with respect to the Earth's source, like
        # you would specify it in a geodetic format, while
        # the lon/lat is specified in J2000, like you would in ECI.
        # Eventually everything should be in ECI (GCRS in astropy
        # for all purposes), but for now let's do the conversion.
        # 1. Get the direction in galactic
        # 2. Transform to GCRS, which uses RA/Dec (ICRS-like).
        #    This is represented in the unit sphere
        # 3. Add the altitude by transforming to EarthLocation.
        #    Should take care of the non-spherical Earth
        # 4. Go back GCRS, now with the correct distance
        #    (from the Earth's center)
        zenith_gal = SkyCoord(l=earth_lon * u.deg, b=earth_lat * u.deg, frame="galactic")
        gcrs = zenith_gal.transform_to('gcrs')
        earth_loc = EarthLocation.from_geodetic(lon=gcrs.ra, lat=gcrs.dec, height=altitude*u.km)

        return cls(time, attitude, earth_loc, livetime)

    def _interp_attitude(self, points, weights) -> Attitude:
        """

        Parameters
        ----------
        points
        weights

        Returns
        -------

        """

        # TODO: we could do a better interpolation using more points, or
        #   additional ACS data e.g. the rotation speed

        rot_matrix = self._attitude.as_matrix()

        interp_attitude = Attitude.from_matrix(rot_matrix[points[0]]*weights[0] + rot_matrix[points[1]]*weights[1], frame = self._attitude.frame)

        return interp_attitude

    def interp_attitude(self, time) -> Attitude:
        """

        Returns
        -------

        """

        points, weights = self.interp_weights(time)

        return self._interp_attitude(points, weights)

    def _interp_location(self, points, weights) -> EarthLocation:
        """

        Parameters
        ----------
        points
        weights

        Returns
        -------

        """

        # TODO: we could do a better interpolation using more points and orbital dynamics

        x = self._location.x
        y = self._location.y
        z = self._location.z

        x_interp = x[points[0]] * weights[0] + x[points[1]] * weights[1]
        y_interp = y[points[0]] * weights[0] + y[points[1]] * weights[1]
        z_interp = z[points[0]] * weights[0] + z[points[1]] * weights[1]

        interp_location = EarthLocation.from_geocentric(x=x_interp, y=y_interp, z=z_interp)

        return interp_location

    def interp_location(self, time) -> EarthLocation:
        """

        Returns
        -------
        """

        points, weights = self.interp_weights(time)

        return self._interp_location(points, weights)

    def _cumulative_livetime(self, points, weights) -> u.Quantity:

        cum_livetime_discrete = np.append(0 * self._hist.unit, np.cumsum(self.livetime))

        up_to_tstart = cum_livetime_discrete[points[0]]

        within_bin = self.livetime[points[0]] * weights[1]

        cum_livetime = up_to_tstart + within_bin

        return cum_livetime

    def cumulative_livetime(self, time: Time) -> u.Quantity:
        """
        Get the cumulative live obstime up to this obstime.

        The live obstime in between the internal timestamp is
        assumed constant.

        Parameters
        ----------
        time:
            Timestamps

        Returns
        -------
        Cummulative live obstime, with units.
        """

        points, weights = self.interp_weights(time)

        return self._cumulative_livetime(points, weights)

    def interp_weights(self, times: Time):
        return self._hist.axis.interp_weights_edges(times)

    def interp(self, times: Time) -> 'SpacecraftFile':

        """
        Linearly interpolates attitude and position at a given obstime

        Parameters
        ----------
        times:
            Timestamps to interpolate

        Returns
        -------
        A new SpacecraftFile object interpolated at these location
        """

        if times.size < 2:
            raise ValueError("We need at least two obstime stamps. See also interp_attitude and inter_location")

        points, weights = self.interp_weights(times)

        interp_attitude = self._interp_attitude(points, weights)
        interp_location = self._interp_location(points, weights)

        cum_livetime = self._cumulative_livetime(points, weights)
        diff_livetime = cum_livetime[1:] - cum_livetime[:-1]

        return self.__class__(times, interp_attitude, interp_location, diff_livetime)

    def select_interval(self, start:Time = None, stop:Time = None) -> "SpacecraftFile":
        """
        Returns the SpacecraftFile file class object for the source interval.

        Parameters
        ----------
        start : astropy.time.Time
            The start obstime of the orientation period. Start of history by default.
        stop : astropy.time.Time
            The end obstime of the orientation period. End of history by default.

        Returns
        -------
        cosipy.spacecraft.SpacecraftFile
        """

        if start is None:
            start = self.tstart

        if stop is None:
            stop = self.tstop

        if start < self.tstart or stop > self.tstop:
            raise ValueError(f"Input range ({start}-{stop}) is outside the SC history ({self.tstart}-{self.tstop})")

        start_points, start_weights = self.interp_weights(start)
        stop_points, stop_weights = self.interp_weights(stop)

        start_attitude = self._interp_attitude(start_points, start_weights)
        stop_attitude = self._interp_attitude(stop_points, stop_weights)

        att_rot = self._attitude.as_matrix()
        new_attitude = Attitude.from_matrix(np.append(start_attitude.as_matrix()[None],
                                                      np.append(att_rot[start_points[1]:stop_points[1]],
                                                                stop_attitude.as_matrix()[None], axis = 0), axis = 0),
                                            frame = self._attitude.frame)

        start_location = self._interp_location(start_points, start_weights)[None]
        stop_location = self._interp_location(stop_points, stop_weights)[None]
        center_locations = self._location[start_points[1]:stop_points[1]]

        new_location = EarthLocation.from_geocentric(np.concatenate((start_location.x, center_locations.x, stop_location.x)),
                                                     np.concatenate((start_location.y, center_locations.y, stop_location.y)),
                                                     np.concatenate((start_location.z, center_locations.z, stop_location.z)))

        first_livetime = self.livetime[start_points[0]]*start_weights[1]
        last_livetime = self.livetime[stop_points[0]]*stop_weights[1]

        new_livetime = np.append(first_livetime, np.append(self.livetime[start_points[1]:stop_points[0]], last_livetime))

        middle_times = self.obstime[start_points[1]:stop_points[1]]

        new_time = Time(np.concatenate(([start.jd1], middle_times.jd1, [stop.jd1])),
                        np.concatenate(([start.jd2], middle_times.jd2, [stop.jd2])),
                        format = 'jd')

        return self.__class__(new_time, new_attitude, new_location, new_livetime)


    def get_target_in_sc_frame(self, target_coord: SkyCoord) -> SkyCoord:

        """
        Get the location in spacecraft coordinates for a given target
        in inertial coordinates.

        Parameters
        ----------
        target_coord : astropy.coordinates.SkyCoord
            The coordinates of the target object.

        Returns
        -------
        astropy.coordinates.SkyCoord
            The target coordinates in the spacecraft frame.
        """

        logger.info("Now converting to the Spacecraft frame...")

        src_path = SkyCoord(np.dot(self.attitude.rot.inv().as_matrix(), target_coord.cartesian.xyz.value),
                            representation_type = 'cartesian',
                            frame = SpacecraftFrame())

        src_path.representation_type = 'spherical'

        return src_path

    def get_dwell_map(self, target_coord:SkyCoord, nside:int, scheme = 'ring') -> HealpixMap:

        """
        Generates the dwell obstime map for the source.

        Parameters
        ----------
        target_coord:
            Source coordinate
        nside:
            Healpix NSIDE
        scheme:
            Healpix pixel ordering scheme

        Returns
        -------
        mhealpy.containers.healpix_map.HealpixMap
            The dwell obstime map.
        """

        # Get source path
        src_path_skycoord = self.get_target_in_sc_frame(target_coord)

        # Empty map
        dwell_map = HealpixMap(nside = nside,
                               scheme = scheme,
                               coordsys = SpacecraftFrame())

        # Fill
        # Get the unique pixels to weight, and sum all the correspondint weights first, so
        # each pixels needs to be called only once.
        # Based on https://stackoverflow.com/questions/23268605/grouping-indices-of-unique-elements-in-numpy

        # remove the last value. Effectively a 0th order interpolations
        pixels, weights = dwell_map.get_interp_weights(theta=src_path_skycoord[:-1])

        weighted_duration = weights * self.livetime.to_value(u.second)[None]

        pixels = pixels.flatten()
        weighted_duration = weighted_duration.flatten()

        pixels_argsort = np.argsort(pixels)

        pixels = pixels[pixels_argsort]
        weighted_duration = weighted_duration[pixels_argsort]

        first_unique = np.concatenate(([True], pixels[1:] != pixels[:-1]))

        pixel_unique = pixels[first_unique]

        splits = np.nonzero(first_unique)[0][1:]
        pixel_durations = [np.sum(weighted_duration[start:stop]) for start, stop in
                           zip(np.append(0, splits), np.append(splits, pixels.size))]

        for pix, dur in zip(pixel_unique, pixel_durations):
            dwell_map[pix] += dur

        dwell_map.to(u.second, update=False, copy=False)

        return dwell_map

    def get_scatt_map(self,
                       nside,
                       target_coord=None,
                       scheme = 'ring',
                       coordsys = 'galactic',
                       r_earth = 6378.0,
                       earth_occ = True
                       ):

        """
        Bin the spacecraft attitude history into a 4D histogram that 
        contains the accumulated obstime the axes of the spacecraft where
        looking at a given direction. 

        Parameters
        ----------
        target_coord : astropy.coordinates.SkyCoord, optional
            The coordinates of the target object. 
        nside : int
            The nside of the scatt map.
        scheme : str, optional
            The scheme of the scatt map (the default is "ring")
        coordsys : str, optional
            The coordinate system used in the scatt map (the default is "galactic).
        r_earth : float, optional
            Earth radius in km (default is 6378 km).
        earth_occ : bool, optional
            Option to include Earth occultation in scatt map calculation.
            Default is True. 

        Returns
        -------
        h_ori : cosipy.spacecraftfile.scatt_map.SpacecraftAttitudeMap
            The spacecraft attitude map.
        """
        
        # Check if target_coord is needed
        if earth_occ and target_coord is None:
            raise ValueError("target_coord is needed when earth_occ = True")

        # Get orientations
        timestamps = self.obstime
        attitudes = self.attitude

        # Altitude at each point in the orbit:
        altitude = self._location.height

        # Earth zenith at each point in the orbit:
        earth_zenith = self.location.itrs

        # Fill (only 2 axes needed to fully define the orientation)
        h_ori = SpacecraftAttitudeMap(nside = nside,
                                      scheme = scheme,
                                      coordsys = coordsys)
        
        x,y,z = attitudes[:-1].as_axes()
       
        # Get max angle based on altitude:
        max_angle = np.pi - np.arcsin(r_earth/(r_earth + altitude))
        max_angle *= (180/np.pi) # angles in degree

        # Define weights and set to 0 if blocked by Earth:
        weight = self.livetime*u.s

        if earth_occ:
            # Calculate angle between source direction and Earth zenith
            # for each obstime stamp:
            src_angle = target_coord.separation(earth_zenith)

            # Get pointings that are occulted by Earth:
            earth_occ_index = src_angle.value >= max_angle

            # Mask
            weight[earth_occ_index[:-1]] = 0
        
        # Fill histogram:
        h_ori.fill(x, y, weight = weight)

        return h_ori



