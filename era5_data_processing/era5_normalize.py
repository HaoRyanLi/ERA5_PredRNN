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

era5_files = sorted(glob('*_3_24hr.npz'))

for filei in era5_files:
    print(filei)
    ds = np.load(filei)
    data = ds['input_raw_data']
    print(data.shape)
    if np.unique(data[20,1,:,:])[10] > 100:
        print('need to fix')
        os.remove(filei)
        data[:,1,:,:] = (data[:,1,:,:] - 98000) / (105000 - 98000)
        final_ds = {}
        final_ds['input_raw_data'] = data
        final_ds['dims'] = np.array([[3,720,1440]]).astype(np.int32)
        final_ds['clips'] = ds['clips']
        np.savez(filei, **final_ds)

