import os
import sys
from time import sleep
#from osgeo import gdal
import numpy as np
from glob import glob
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import pygrib
from collections import defaultdict

data_path = '/scratch/09012/haoli1/ERA5/'
dataset_path = '/scratch/09012/haoli1/ERA5/dataset/'
infile = sys.argv[1]
print(f"infile: {infile}")

data = pygrib.open(data_path+infile)
for ii,g in enumerate(data):
    if ii >1:
        break
    print(ii, str(g.name), g.date)
    if ii % 2 == 0:
        Geopotential = np.array(g.values).astype(np.float32)
    elif ii % 2 == 1:
        Land_sea_mask = np.array(g.values).astype(np.float32)

def int_721_to_720(variables):
    # var_int = (variables[:,1:] - variables[:,:-1])/dx*lat_diff[None,:,:] + variables[:,:-1]
    var_int = (variables[1:] + variables[:-1])*0.5
    return var_int

Geopotential = int_721_to_720(Geopotential)
Land_sea_mask = int_721_to_720(Land_sea_mask)

norm_dict = {'geopoential':[0,50000],
            'land':[0,1]}

Geopotential = (Geopotential - norm_dict['geopoential'][0])/(norm_dict['geopoential'][1] - norm_dict['geopoential'][0])


print(f"Geopotential shape: {Geopotential.shape}, max: {np.max(Geopotential)}")
print(f"Land_sea_mask shape: {Land_sea_mask.shape}, max: {np.max(Land_sea_mask)}")

extra_var = {}
extra_var['geopoential'] = Geopotential
extra_var['land'] = Land_sea_mask

np.savez(dataset_path+'exta_var.npz', **extra_var)
