import numpy as np
from astropy.time import Time
from astropy.io import fits

class GoodTimeInterval():
    
    def __init__(self, tstart_list, tstop_list):
        """
        Initialize GTI object.
        
        Parameters
        ----------
        tstart_list : astropy.time.Time (array)
            Start times of GTI intervals
        tstop_list : astropy.time.Time (array)
            Stop times of GTI intervals
        """
        # Check that starts and stops are scalar
        if tstart_list.isscalar == True:
            tstart_list = Time([tstart_list])
        if tstop_list.isscalar == True:
            tstop_list = Time([tstop_list])

        # Check that starts and stops have the same scale
        if not np.all(tstart_list.scale == tstop_list.scale):
            raise ValueError(f"Time scale mismatch between starts ({tstart_list.scale}) and stops ({tstop_list.scale})")
        
        # Check that starts and stops have the same format
        if tstart_list.format != tstop_list.format:
            raise ValueError(f"Time format mismatch between starts ({tstart_list.format}) and stops ({tstop_list.format})")
        
        # Check that starts and stops have the same length 
        if len(tstart_list) != len(tstop_list):
            raise ValueError(f"Length mismatch between starts ({len(tstart_list)}) and stops ({len(tstop_list)})")
        
        self._tstart_list = tstart_list
        self._tstop_list = tstop_list
        
        # Sort by start time
        self.sort()

    @property
    def tstart_list(self):
        return self._tstart_list
    
    @property
    def tstop_list(self):
        return self._tstop_list
    
    def __len__(self):
        """Return the number of GTI intervals."""
        return len(self._tstart_list)
    
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
            (tstart_list, tstop_list) for the indexed interval(s)
        """
        return self._tstart_list[index], self._tstop_list[index]
    
    def __iter__(self):
        """
        Iterate over GTI intervals.
        
        Yields
        ------
        tuple of (Time, Time)
            Each (start, stop) pair
        """
        for start, stop in zip(self._tstart_list, self._tstop_list):
            yield start, stop
        
    def sort(self):
        """
        Sort GTI by start time in ascending order.
        
        Modifies the GTI in place.
        Stops are sorted according to the start time order.
        """
        sort_idx = np.argsort(self._tstart_list)
        self._tstart_list = self._tstart_list[sort_idx]
        self._tstop_list = self._tstop_list[sort_idx]
    
    def save_as_fits(self, filename, overwrite=False, output_format='unix'):
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
        """
        # Get values in the specified output format using getattr
        if not hasattr(self._tstart_list, output_format):
            raise ValueError(f"Unsupported output format: {output_format}")
        
        start_times = getattr(self._tstart_list, output_format)
        stop_times = getattr(self._tstop_list, output_format)
        
        # Use the scale from the stored Time objects
        output_scale = self._tstart_list.scale
        
        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['TIMESYS'] = output_scale.upper()
        output_unit = 's'
        if output_format in ['jd', 'mjd']:
            output_unit = 'd'
        primary_hdu.header['TIMEUNIT'] = output_unit
        
        # Define table columns
        col1 = fits.Column(name='TSTART', format='D', unit=output_unit, array=start_times)
        col2 = fits.Column(name='TSTOP', format='D', unit=output_unit, array=stop_times)
        
        # Create table HDU
        table_hdu = fits.BinTableHDU.from_columns([col1, col2])
        table_hdu.header['EXTNAME'] = 'GTI'
        table_hdu.header['TIMESYS'] = output_scale.upper()
        table_hdu.header['TIMEUNIT'] = output_unit
        table_hdu.header['TIMEFORMAT'] = output_format
        
        # Create HDUList and write to FITS file
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(filename, overwrite=overwrite)
    
    @classmethod
    def from_fits(cls, filename):
        """
        Load GTI from a FITS file.
        
        Reads time format and scale from FITS header.
        
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
        
        # Read time system/format from header
        time_scale = gti_hdu.header.get('TIMESYS', 'utc').lower()
        time_format = gti_hdu.header.get('TIMEFORMAT', 'unix').lower()
        
        # Read start and stop times as arrays
        tstart_list = Time(gti_hdu.data['TSTART'], format=time_format, scale=time_scale)
        tstop_list = Time(gti_hdu.data['TSTOP'], format=time_format, scale=time_scale)
        
        infile.close()
        return cls(tstart_list, tstop_list)
