import os
from time import sleep
import numpy as np
from glob import glob
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import pygrib
import numpy
import matplotlib
import torch
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
from collections import defaultdict

path = '/scratch/09012/haoli1/ERA5/'
data_01_02 = pygrib.open(path+'download_2014_2020_01_02.grib')

d = defaultdict(int)
d['u_wind_arr'] = 0
d['v_wind_arr'] = 0
d['sea_press_arr'] = 0
d['precip_arr'] = 0
d['temp_2m_arr'] = 0

curr_year ='2015'

i_sum = 0

for ii,g in enumerate(data_01_02):
    if ii <=126:
        print("Index:", ii, "Date:", g.date, "Time:", g.time, "Name:", g.name)
    if curr_year != str(g.date)[:4]:
        if int(curr_year) < int(str(g.date)[:4]):
            break
        else:
            continue
    
    # print("Index:", ii, "Date:", g.date, "Time:", g.time, "Name:", g.name)
    i_sum += ii
    if ii % 5 == 0:
        if g.name == '10 metre U wind component':
            d['u_wind_arr'] += 1
        else:
            print("Error:", ii, str(g.name))
    elif ii % 5 == 1: 
        if g.name == '10 metre V wind component':
            d['v_wind_arr'] += 1
        else:
            print("Error:", ii, str(g.name))
    elif ii % 5 == 2: 
        if g.name == 'Mean sea level pressure':
            d['sea_press_arr'] += 1
        else:
            print("Error:", ii, str(g.name))
    elif ii % 5 == 3:
        d['precip_arr'] += 1
        # if g.name == 'Total precipitation':
        #     d['precip_arr'] += 1
        # else:
        #     print("Error:", ii, str(g.name))
    elif ii % 5 == 4: 
        if g.name == '2 metre temperature':
            d['temp_2m_arr'] += 1
        else:
            print("Error:", ii, str(g.name))

for key, val in d.items():
    print(key, val)

print(i_sum)