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

infile = sys.argv[1]

data_ds1 = pygrib.open(infile)

curr_year = int(sys.argv[2])
curr_month = 9
u_wind_arr1 = []
v_wind_arr1 = []
sea_press_arr1 = []
precip_arr1 = []

for ii,g in enumerate(data_ds1):
    if str(curr_year) != str(g.date)[:4]:
        continue
    if '10 metre U wind component' == str(g.name):
        u_wind_arr1.append(g.values[np.newaxis,...])
    elif '10 metre V wind component' == str(g.name):
        v_wind_arr1.append(g.values[np.newaxis,...])
    elif 'Mean sea level pressure' == str(g.name):
        sea_press_arr1.append(g.values[np.newaxis,...])
    elif 'Total precipitation' == str(g.name):
        precip_arr1.append(g.values[np.newaxis,...])
    else:
        continue

print(len(precip_arr1))

norm_dict = {'u_wind':[-20,20],
            'v_wind':[-20,20],
            'temp':[210,305],
            'sea_press':[98000,105000],
            'precip':[-7,-2],
            'vapor':[-1,2.]}

stack_u_wind = np.vstack(u_wind_arr1)/20.
stack_v_wind = np.vstack(v_wind_arr1)/20.
stack_press = np.vstack(sea_press_arr1)
stack_precip = np.vstack(precip_arr1)
stack_precip[stack_precip==0] = 1e-9
stack_precip = (np.log10(stack_precip) - norm_dict['precip'][0])/(norm_dict['precip'][1] - norm_dict['precip'][0])

wind_speed = np.sqrt((stack_u_wind)**2+(stack_v_wind)**2)

final_data = np.vstack([wind_speed[np.newaxis,...],stack_press[np.newaxis,...],stack_precip[np.newaxis,...]])
final_data = np.swapaxes(final_data,0,1)

final_data = final_data[:,:,:720,:]
#final_data[:,2,:,:] = final_data[:,2,:,:] - 0.5
#final_data[:,5,:,:] = final_data[:,5,:,:] - 0.5
n_step = final_data.shape[0]
print(n_step)
print(final_data.shape)

in_step = 24
out_step = 24

final_clips = np.ones((2,int(np.ceil(n_step/(in_step))),2))*in_step
final_clips[0,:,0] = np.arange(0,n_step,in_step)

final_clips[1,:,0] = np.arange(in_step,n_step+1,out_step)
final_clips[1,:,1] = out_step
print(final_clips.shape)

final_ds = {}
final_ds['input_raw_data'] = final_data
final_ds['dims'] = np.array([[3,720,1440]]).astype(np.int32)
final_ds['clips'] = final_clips.astype(np.int32)

dataset_path = '/scratch/09012/haoli1/ERA5/hoang_dataset/'
np.savez(dataset_path+'era5_train_'+str(curr_month).zfill(2)+'01'+str(curr_year)+'_3_24hr.npz', **final_ds)

