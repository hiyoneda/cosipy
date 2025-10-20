import numpy as np
from astropy.time import Time
from astropy.io import fits

class GoodTimeInterval():
    
    def __init__(self, starts, stops):
        """
        Initialize GTI object.
        
        Parameters
        ----------
        starts : astropy.time.Time (array)
            Start times of GTI intervals
        stops : astropy.time.Time (array)
            Stop times of GTI intervals

        Notes
        -----
        Currently, unix + utc is assumed. 
        When the default time format/system is fixed,
        this class should be modified.
        """
        # Check that starts and stops have the same scale
        if not np.all(starts.scale == stops.scale):
            raise ValueError(f"Time scale mismatch between starts ({starts.scale}) and stops ({stops.scale})")
        
        # Check that starts and stops have the same format
        if starts.format != stops.format:
            raise ValueError(f"Time format mismatch between starts ({starts.format}) and stops ({stops.format})")
        
        self.starts = starts
        self.stops = stops
        
        # Sort by start time
        self.sort()
    
    def __len__(self):
        """Return the number of GTI intervals."""
        return len(self.starts)
    
    def __getitem__(self, index):
        """
        Get GTI interval(s) by index.
        
        Parameters
        ----------
        index : int, slice, or array-like
            Index, slice, or boolean/integer array to retrieve
        
        Returns
        -------
        tuple of (Time, Time)
            (starts, stops) for the indexed interval(s)
        """
        return self.starts[index], self.stops[index]
    
    def __iter__(self):
        """
        Iterate over GTI intervals.
        
        Yields
        ------
        tuple of (Time, Time)
            Each (start, stop) pair
        """
        for start, stop in zip(self.starts, self.stops):
            yield start, stop
        
    def sort(self):
        """
        Sort GTI by start time in ascending order.
        
        Modifies the GTI in place.
        Stops are sorted according to the start time order.
        """
        sort_idx = np.argsort(self.starts)
        self.starts = self.starts[sort_idx]
        self.stops = self.stops[sort_idx]
    
    def is_in_gti(self, time):
        """
        Check if a time (or list of times) is within any GTI interval.
        
        Uses binary search for efficiency, assuming GTI is sorted.
        
        Parameters
        ----------
        time : astropy.time.Time
            Time or times to check (scalar or array)
            Must be in the same time scale as the GTI.
        
        Returns
        -------
        bool or numpy.ndarray of bool
            True if time is within GTI, False otherwise.
            If input is array, returns array of booleans.
        int or numpy.ndarray of int
            Index of the GTI interval containing the time(s).
            -1 if not in any GTI interval.
        """
        # Check time scale
        if time.scale != self.starts.scale:
            raise ValueError(f"Time scale mismatch. Expected {self.starts.scale.upper()}, "
                           f"got {time.scale.upper()}")
        
        # Get values using the format attribute
        time_format = self.starts.format
        starts_value = getattr(self.starts, time_format)
        stops_value = getattr(self.stops, time_format)
        times_value = getattr(time, time_format)
        
        # Check if time is scalar or array
        if time.isscalar:
            # Single time
            idx = np.searchsorted(starts_value, times_value, side='right') - 1
            if idx >= 0 and idx < len(stops_value):
                if times_value <= stops_value[idx]:
                    return True, idx
            return False, -1
        else:
            # Array of times - vectorized with np.searchsorted
            indices = np.searchsorted(starts_value, times_value, side='right') - 1
            
            # Check validity and whether times fall within GTI intervals
            valid = (indices >= 0) & (indices < len(stops_value))
            result = np.zeros(len(time), dtype=bool)
            result[valid] = times_value[valid] <= stops_value[indices[valid]]
            indices[~result] = -1
            
            return result, indices
    
    def save_as_fits(self, filename, overwrite=False, output_format='unix', output_unit='s'):
        """
        Save GTI data to a FITS file.
        
        Parameters
        ----------
        filename : str
            Output FITS filename
        overwrite : bool, optional
            If True, overwrite existing file (default: False)
        output_format : str, optional
            Time format for output (e.g., 'unix', 'mjd'). Default: 'unix'
        output_unit : str, optional
            Time unit for output. Default: 's'
        """
        # Get values in the specified output format using getattr
        if not hasattr(self.starts, output_format):
            raise ValueError(f"Unsupported output format: {output_format}")
        
        start_times = getattr(self.starts, output_format)
        stop_times = getattr(self.stops, output_format)
        
        # Use the scale from the stored Time objects
        output_scale = self.starts.scale
        
        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['TIMESYS'] = output_scale.upper()
        primary_hdu.header['TIMEUNIT'] = output_unit
        
        # Define table columns
        col1 = fits.Column(name='TSTART', format='D', unit=output_unit, array=start_times)
        col2 = fits.Column(name='TSTOP', format='D', unit=output_unit, array=stop_times)
        
        # Create table HDU
        table_hdu = fits.BinTableHDU.from_columns([col1, col2])
        table_hdu.header['EXTNAME'] = 'GTI'
        table_hdu.header['TIMESYS'] = output_scale.upper()
        table_hdu.header['TIMEUNIT'] = output_unit
        
        # Create HDUList and write to FITS file
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(filename, overwrite=overwrite)
    
    @classmethod
    def from_fits(cls, filename):
        """
        Load GTI from a FITS file.
        
        Reads time format and scale from FITS header.
        Currently supports UNIX time format.
        TODO: Add support for MJD, MET (MJDREFI/MJDREFF) formats.
        
        Parameters
        ----------
        filename : str
            Input FITS filename
            
        Returns
        -------
        GoodTimeIntervals
            GTI object
        """
        infile = fits.open(filename)
        
        # Search for GTI extension
        gti_hdu = None
        for hdu in infile:
            if isinstance(hdu, fits.BinTableHDU) and hdu.name in ['GTI']:
                gti_hdu = hdu
                break
        
        if gti_hdu is None:
            infile.close()
            raise ValueError("GTI table not found in FITS file")
        
        # Read time system from header
        time_scale = gti_hdu.header.get('TIMESYS', 'UTC').lower()
        time_unit = gti_hdu.header.get('TIMEUNIT', 's')
        
        # TODO: Auto-detect time format from header or data
        # For now, assume UNIX time
        time_format = 'unix'
        
        # Read start and stop times as arrays
        starts = Time(gti_hdu.data['TSTART'], format=time_format, scale=time_scale)
        stops = Time(gti_hdu.data['TSTOP'], format=time_format, scale=time_scale)
        
        infile.close()
        return cls(starts, stops)
