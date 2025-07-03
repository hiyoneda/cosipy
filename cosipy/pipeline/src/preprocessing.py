from cosipy.pipeline.src.io import load_binned_data
from astropy.time import Time
from astropy.io.misc import yaml
from cosipy.response import FullDetectorResponse
from cosipy import BinnedData

import numpy as np

def write_yaml(udata_path,ori_path,resp_path,dt,tmin,tmax,bin_yaml_path):
    with FullDetectorResponse.open(resp_path) as response:
        bin_dict={
            "data_file": udata_path,
            "ori_file": ori_path,
            "unbinned_output": udata_path[-4:],
            "time_bins": dt,
            "energy_bins": list(response.axes["Em"].edges.value),
            "phi_pix_size": int(180/(response.axes["Phi"].nbins)),
            "nside": response.nside,
            "scheme": response.scheme,
            "tmin": tmin,
            "tmax": tmax
        }
        with open(bin_yaml_path, 'w') as outfile:
            yaml.dump(bin_dict, outfile, default_flow_style=False, sort_keys=False)
    return(bin_dict)



def get_binned_data(yaml_path, udata_path, bdata_name, psichi_coo):
    """
    Creates a binned dataset from a yaml file and an unbinned data file.
    Args:
        psichi_coo: either "galactic" or "local"

    """
    data=BinnedData(yaml_path)
    data.get_binned_data(unbinned_data=udata_path, output_name=bdata_name, psichi_binning=psichi_coo,make_binning_plots=False)
    return data




def tslice_binned_data(data,tmin,tmax):
    """Slice a binned dataset in time"""
    idx_tmin = np.where(data.axes['Time'].edges.value >= tmin.value)[0][0]
    idx_tmax_all = np.where(data.axes['Time'].edges.value <= tmax.value)
    y = len(idx_tmax_all[0]) - 1
    idx_tmax = np.where(data.axes['Time'].edges.value <= tmax.value)[0][y]
    tsliced_data = data.slice[{'Time': slice(idx_tmin, idx_tmax)}]
    return tsliced_data



def tslice_ori(ori,tmin,tmax):
    """
    Slices time for the orientation file
    """
    #ori_min = Time(tmin,format = 'unix')
    #ori_max = Time(tmax,format = 'unix')
    tsliced_ori = ori.source_interval(tmin, tmax)
    return tsliced_ori